"""Experiment entry point: data, train, evaluate, plot, and an end-to-end CPU smoke."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
from data import Config, POLICIES, DISKS, make_dataset, trace, write_json
from env import Env

HEURISTICS = ("Greedy_latency", "Image_priority", "Least_load", "Greedy_resources", "Random")


def load_data(root, split):
    root = Path(root)
    catalogue = json.loads((root / "world.json").read_text())
    requests = [json.loads(p.read_text()) for p in sorted((root / split).glob("*.json"))]
    if not requests:
        raise ValueError(f"No {split} traces in {root}; run the data command first")
    return catalogue, requests


def fingerprint(catalogue):
    return hashlib.sha256(json.dumps(catalogue, sort_keys=True).encode()).hexdigest()


def rollout(catalogue, requests, policy, disk, method="Agent", agent=None, seed=0,
            explore=False, collect=False, record=False, replay_slots=200, reward_mode="potential"):
    if reward_mode not in ("potential", "holding", "completion"):
        raise ValueError("Unknown reward mode")
    env = Env(catalogue, requests, policy, disk, record, replay_slots)
    rng = np.random.default_rng(seed)
    history_size = agent.history if agent else 1
    history = np.zeros((history_size, *env.obs_shape), np.float32)
    states, masks, actions, rewards = [], [], [], []
    potential = 0.
    while env.task:
        obs, mask = env.observation(), env.mask()
        history[:-1] = history[1:]
        history[-1] = obs
        action = agent.act(history, mask, rng if explore else None) if agent else env.heuristic(method, rng)
        if collect:
            states.append(obs.astype(np.float16)); masks.append(mask); actions.append(action)
        allowance = env.task["deadline"] - env.task["arrival"]
        previous_cost = env.latency_cost
        env.step(action)
        if collect:
            reward = (allowance - (env.latency_cost - previous_cost)) / 50
            if reward_mode == "potential":
                next_potential = -env.remaining_cost() / 50 if env.task else 0.
                reward += .98 * next_potential - potential
                potential = next_potential
            rewards.append(reward)
    experience = None
    if collect:
        states.append(np.zeros(env.obs_shape, np.float16))
        terminal_mask = np.zeros(len(env.nodes) + 1, bool)
        terminal_mask[-1] = True
        masks.append(terminal_mask)
        if reward_mode == "completion":
            rewards = [(env.jobs[i]["deadline"] - env.jobs[i]["finish"]) / 50 for i in range(len(actions))]
        experience = states, masks, actions, rewards
    return env, experience


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train(args):
    import jax
    from agent import SAC, Replay
    catalogue, training = load_data(args.data, "train")
    _, validation = load_data(args.data, "val")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(command="train", arguments=vars(args), world_sha256=fingerprint(catalogue),
                    python=platform.python_version(), numpy=np.__version__, jax=jax.__version__,
                    devices=[str(d) for d in jax.devices()], smoke=args.smoke_run,
                    source_sha256={f: hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest()
                                   for f in ("data.py", "env.py", "agent.py", "run.py")},
                    note="Smoke validates mechanics only; convergence requires full multi-seed training.")
    write_json(output / "train_manifest.json", manifest)
    for policy in args.policies:
        for disk in args.disks:
            baseline_path = output / "validation_baselines.csv"
            existing_baselines = []
            if baseline_path.exists():
                with baseline_path.open() as f:
                    existing_baselines = list(csv.DictReader(f))
            if not any(r["policy"] == policy and r["disk"] == disk for r in existing_baselines):
                for request in validation:
                    env, _ = rollout(catalogue, request, policy, disk, method="Greedy_latency")
                    append_csv(baseline_path, dict(policy=policy, disk=disk, trace=request["id"], **env.metrics()))
            for seed in args.seeds:
                name = f"{policy}_{disk}_s{seed}"
                folder = output / "models" / name
                continuing = args.resume and (folder / "last.pkl").exists()
                if args.resume and not continuing:
                    raise FileNotFoundError(f"No last.pkl to resume in {folder}")
                if folder.exists() and any(folder.iterdir()) and not continuing:
                    if args.skip_existing and (folder / "complete.json").exists():
                        print(f"Skipping completed {name}", flush=True)
                        continue
                    raise FileExistsError(f"{folder} already contains a run; use a new --output or --skip-existing for completed runs")
                folder.mkdir(parents=True, exist_ok=True)
                agent = SAC.load(folder / "last.pkl") if continuing else SAC(Env(catalogue, training[0]).obs_shape, seed, args.width, args.history, local_branch=args.local_branch)
                if continuing:
                    expected = dict(policy=policy, disk=disk, seed=seed, world_sha256=fingerprint(catalogue))
                    if any(agent.metadata[k] != value for k, value in expected.items()):
                        raise ValueError("Resume checkpoint does not match the requested experiment")
                    if (agent.width, agent.history) != (args.width, args.history):
                        raise ValueError("Use the checkpoint's --width and --history when resuming")
                    if bool(agent.metadata["smoke"]) != args.smoke_run:
                        raise ValueError("Smoke and scientific training runs cannot be mixed")
                    if args.local_branch and "local" not in agent.actor:
                        raise ValueError("Cannot add --local-branch to an existing checkpoint; start a separate run")
                    if agent.metadata.get("reward_mode", "completion") != args.reward_mode:
                        raise ValueError("Resume with the checkpoint's --reward-mode; older checkpoints use completion")
                first_episode = int(agent.metadata["episode"]) + 1 if continuing else 0
                if first_episode > args.episodes:
                    print(f"Skipping {name}: saved episode {first_episode - 1} already reaches target {args.episodes}", flush=True)
                    continue
                replay = Replay(args.replay_capacity, args.history)
                rng = np.random.default_rng(seed)
                order = np.arange(len(training))
                if continuing and "sampler_state" in agent.metadata:
                    rng.bit_generator.state = agent.metadata["sampler_state"]
                    order = np.asarray(agent.metadata["trace_order"])
                best = SAC.load(folder / "best.pkl").metadata["validation"]["reward"] if continuing else -np.inf
                # Identical warm-start trace IDs across configurations; train split only.
                for i in range(args.warmup_episodes):
                    _, exp = rollout(catalogue, training[i % len(training)], policy, disk,
                                     method="Greedy_latency", agent=agent if continuing else None,
                                     explore=continuing, seed=seed + i, collect=True, reward_mode=args.reward_mode)
                    replay.add(*exp)
                for _ in range(args.bc_updates if replay.size and not continuing else 0):
                    agent.update(replay.sample(args.batch_size, rng), imitation=True)
                if continuing:
                    print(f"Resuming {name} at episode {first_episode}; weights/optimizers retained, replay rebuilt from training traces", flush=True)
                started = time.monotonic()
                for episode in range(first_episode, args.episodes + 1):
                    diagnostics = dict(actor_loss=0., critic_loss=0., alpha=float(np.exp(agent.log_alpha)), entropy=0., target_q=0., grad_norm=0.)
                    if episode:
                        # Rotate stored training traces with seeded permutations. No test feedback.
                        if (episode - 1) % len(training) == 0:
                            order = rng.permutation(len(training))
                        request = training[order[(episode - 1) % len(training)]]
                        env, exp = rollout(catalogue, request, policy, disk, agent=agent,
                                           seed=int(rng.integers(2**31)), explore=True, collect=True, reward_mode=args.reward_mode)
                        replay.add(*exp)
                        count = max(1, int(len(exp[2]) * args.update_ratio))
                        recent = [agent.update(replay.sample(args.batch_size, rng)) for _ in range(count)]
                        diagnostics = {k: float(np.mean([r[k] for r in recent])) for k in recent[0]}
                        append_csv(output / "training.csv", dict(run=name, policy=policy, disk=disk, seed=seed,
                                   episode=episode, updates=agent.updates, **env.metrics(), **diagnostics))
                        append_csv(output / "signals.csv", dict(run=name, policy=policy, disk=disk, seed=seed,
                                   episode=episode, reward_mode=args.reward_mode,
                                   mean_learning_reward=float(np.mean(exp[3])),
                                   discounted_learning_return=float(np.dot(.98 ** np.arange(len(exp[3])), exp[3]))))
                    if episode % args.eval_every == 0 or episode == args.episodes:
                        evaluations = [rollout(catalogue, request, policy, disk, agent=agent)[0].metrics() for request in validation]
                        average = {k: float(np.mean([e[k] for e in evaluations])) for k in evaluations[0]}
                        append_csv(output / "validation.csv", dict(run=name, policy=policy, disk=disk, seed=seed,
                                   episode=episode, updates=agent.updates, **average))
                        metadata = dict(policy=policy, disk=disk, episode=episode, seed=seed,
                                        validation=average, world_sha256=fingerprint(catalogue),
                                        smoke=args.smoke_run, bc_updates=args.bc_updates, arguments=vars(args),
                                        sampler_state=rng.bit_generator.state, trace_order=order.tolist(),
                                        resumed_from_episode=first_episode - 1 if continuing else None,
                                        replay_rebuilt_on_resume=continuing)
                        metadata["local_branch"] = "local" in agent.actor
                        metadata["reward_mode"] = args.reward_mode
                        if average["reward"] > best:
                            best = average["reward"]
                            agent.save(folder / "best.pkl", metadata)
                        agent.save(folder / "last.pkl", metadata)
                        print(f"{name} episode={episode}/{args.episodes} validation latency={average['avg_latency']:.2f}s "
                              f"on-time={average['completion_ratio']:.1%} hit={average['hit_rate']:.1%} "
                              f"reward={average['reward']:.3f} updates={agent.updates}", flush=True)
                write_json(folder / "complete.json", dict(best_validation_reward=best, episodes=args.episodes,
                           updates=agent.updates, seconds=time.monotonic() - started, smoke=args.smoke_run, arguments=vars(args)))


def learning_info(agent):
    return dict(training_reward=agent.metadata.get("reward_mode", "completion"),
                architecture="CNN+GRU+local" if "local" in agent.actor else "CNN+GRU",
                checkpoint_episode=agent.metadata["episode"])


def evaluate(args):
    from agent import SAC
    catalogue, testing = load_data(args.data, "test")
    output = Path(args.output)
    results = []
    for policy in args.policies:
        for disk in args.disks:
            for seed in args.seeds:
                checkpoint = output / "models" / f"{policy}_{disk}_s{seed}" / "best.pkl"
                agent = SAC.load(checkpoint)
                if agent.metadata["world_sha256"] != fingerprint(catalogue):
                    raise ValueError("Checkpoint and dataset catalogue differ")
                for ti, requests in enumerate(testing):
                    for method in ("Agent", *HEURISTICS):
                        # Deterministic heuristics need only one copy per test trace.
                        if method != "Agent" and seed != args.seeds[0]:
                            continue
                        record = ti == 0 and seed == args.seeds[0]
                        env, _ = rollout(catalogue, requests, policy, disk, method=method,
                                         agent=agent if method == "Agent" else None, seed=seed,
                                         record=record, replay_slots=args.replay_slots)
                        results.append(dict(method=method, policy=policy, disk=disk, seed=seed,
                                            trace=requests["id"], smoke=agent.metadata["smoke"],
                                            **(learning_info(agent) if method == "Agent" else dict(training_reward="", architecture="", checkpoint_episode="")),
                                            **env.metrics()))
                        if record:
                            replay = env.replay(method, seed)
                            if method == "Agent":
                                replay["learning"] = learning_info(agent)
                            write_json(output / "replays" / f"{method}_{policy}_{disk}_{requests['id']}.json", replay)
                print(f"Evaluated {policy}/{disk}/seed {seed}", flush=True)
    path = output / "results.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    # Same frozen actor across all policies and disk regimes for controlled animation.
    if "LRU" in args.policies and "tight" in args.disks:
        agent = SAC.load(output / "models" / f"LRU_tight_s{args.seeds[0]}" / "best.pkl")
        for policy, disk in [(p, "tight") for p in args.policies] + ([("LRU", "large")] if "large" in args.disks else []):
            env, _ = rollout(catalogue, testing[0], policy, disk, agent=agent, record=True, replay_slots=args.replay_slots)
            replay = env.replay("FrozenAgent", args.seeds[0])
            replay["learning"] = learning_info(agent)
            write_json(output / "replays" / f"FrozenAgent_{policy}_{disk}_{testing[0]['id']}.json", replay)
    write_json(output / "evaluation_manifest.json", dict(arguments=vars(args), world_sha256=fingerprint(catalogue),
               test_traces=[r["id"] for r in testing], selection="best mean validation reward; no test selection"))


def smoke(args):
    import subprocess
    import sys
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use an empty --output for smoke to keep artifacts attributable")
    args.data = str(root / "data")
    make_dataset(args.data, Config(cpu_nodes=6, gpu_nodes=3, images=50, slots=40, min_tasks=2, max_tasks=8, phase_slots=12), (3, 1, 1))
    args.policies, args.disks, args.seeds = list(POLICIES), list(DISKS), [0]
    args.episodes, args.width, args.history, args.batch_size = 2, 16, 4, 8
    args.eval_every, args.warmup_episodes, args.bc_updates, args.update_ratio = 1, 1, 4, .03
    args.replay_capacity, args.skip_existing, args.smoke_run = 2000, False, True
    args.resume = False
    args.replay_slots = 40
    train(args)
    evaluate(args)
    from plots import plot_all
    from viz import render_scenarios
    plot_all(root)
    render_scenarios(root, fps=6, slots=16)
    subprocess.run([sys.executable, "-m", "pytest", "-q", "test_core.py"], check=True,
                   env={**os.environ, "MPLCONFIGDIR": "/tmp/ches-matplotlib"})
    write_json(root / "SMOKE_ONLY.json", dict(status="passed", claim="Mechanics verified; these are not converged scientific results."))
    print(f"Smoke complete: {root.resolve()} (not a convergence benchmark)", flush=True)


def record_scenarios(args):
    """Evaluate one frozen checkpoint for controlled replays, without retraining."""
    from agent import SAC
    if not args.checkpoint:
        raise ValueError("The replay command requires --checkpoint")
    catalogue, testing = load_data(args.data, "test")
    agent = SAC.load(args.checkpoint)
    if agent.metadata["world_sha256"] != fingerprint(catalogue):
        raise ValueError("Checkpoint and dataset catalogue differ")
    root = Path(args.output) / "replays"
    for policy, disk in [(p, "tight") for p in POLICIES] + [("LRU", "large")]:
        env, _ = rollout(catalogue, testing[0], policy, disk, agent=agent, record=True, replay_slots=args.replay_slots)
        replay = env.replay("FrozenAgent", agent.seed)
        replay["learning"] = learning_info(agent)
        write_json(root / f"FrozenAgent_{policy}_{disk}_{testing[0]['id']}.json", replay)
    env, _ = rollout(catalogue, testing[0], "LRU", "tight", method="Greedy_latency", record=True, replay_slots=args.replay_slots)
    write_json(root / f"Greedy_latency_LRU_tight_{testing[0]['id']}.json", env.replay("Greedy_latency", agent.seed))
    print(f"Controlled replay logs: {root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("data", "train", "evaluate", "plot", "replay", "smoke"))
    parser.add_argument("--smoke", action="store_true", help="Alias for the smoke command")
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu", "cuda"), default="auto",
                        help="gpu/cuda selects NVIDIA CUDA explicitly; cpu forces CPU; auto lets JAX choose")
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--disks", nargs="+", choices=tuple(DISKS), default=list(DISKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--slots", type=int, default=400)
    parser.add_argument("--train-traces", type=int, default=64)
    parser.add_argument("--val-traces", type=int, default=4)
    parser.add_argument("--test-traces", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--local-branch", action="store_true", help="Learn a direct affine score from per-node immediate features")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--reward-mode", choices=("potential", "holding", "completion"), default="potential",
                        help="Causal holding-time rewards with state-potential shaping (default); holding without shaping; legacy completion")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--warmup-episodes", type=int, default=2)
    parser.add_argument("--bc-updates", type=int, default=100)
    parser.add_argument("--update-ratio", type=float, default=.25)
    parser.add_argument("--replay-capacity", type=int, default=30000)
    parser.add_argument("--replay_slots", "--replay-slots", type=int, default=200)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue last checkpoint; rebuild replay from training traces")
    parser.add_argument("--checkpoint")
    parser.set_defaults(smoke_run=False)
    args = parser.parse_args()
    if args.smoke:
        args.command = "smoke"
    if args.command is None:
        parser.error("Choose a command or --smoke")
    for field in ("slots", "train_traces", "val_traces", "test_traces", "batch_size", "width", "history", "eval_every", "replay_capacity", "replay_slots"):
        if getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.episodes < 0 or args.update_ratio <= 0 or args.warmup_episodes < 0 or args.bc_updates < 0:
        parser.error("Episodes/warmup/BC must be nonnegative; update ratio must be positive")
    if args.device != "auto":
        # In JAX 0.6.2, the 'gpu' alias expands to CUDA AND ROCm and requires both.
        os.environ["JAX_PLATFORMS"] = "cuda" if args.device == "gpu" else args.device
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/ches-matplotlib")
    if args.command == "data":
        root = Path(args.data)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError("Data directory is not empty; choose a fresh --data directory")
        make_dataset(root, Config(slots=args.slots), (args.train_traces, args.val_traces, args.test_traces))
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "plot":
        from plots import plot_all
        plot_all(args.output)
    elif args.command == "replay":
        record_scenarios(args)
    else:
        smoke(args)


if __name__ == "__main__":
    main()
