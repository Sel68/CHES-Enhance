"""CHES + finite disks: data | train | evaluate | plot | smoke.

  python run.py data     --data data
  python run.py train    --data data --out outputs/run --device gpu --episodes 2000
  python run.py evaluate --data data --out outputs/run --device gpu
  python run.py plot     --out outputs/run
  python run.py smoke    --device cpu          # tiny end-to-end check (also renders animations)
"""
import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path
import numpy as np


def setup(device):
    if device != "auto":
        os.environ["JAX_PLATFORMS"] = "cuda" if device == "gpu" else device
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    jax.config.update("jax_compilation_cache_dir", str(Path.home() / ".cache" / "ches_jax"))
    return jax


def fmt_time(s):
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def append_csv(path, rows):
    path = Path(path)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


# ====================================================================== compiled pieces
def build(world, T, args):
    import jax
    import jax.numpy as jnp
    import env as E
    import agent as A
    W = E.to_jax(world)
    N, I = len(world["node_x"]), len(world["img_size"])
    H, hp = args.history, dict(A.HP, width=args.width, gru=args.width, history=args.history, batch=args.batch,
                                gamma=args.gamma, ent_ratio=args.ent_ratio,
                                decompose=not args.plain_critic)
    vview = jax.vmap(E.view, in_axes=(None, None, None, 0, 0))
    vobs = jax.vmap(E.observe, in_axes=(None, None, 0, 0))
    vstep = jax.vmap(E.step, in_axes=(None, None, 0, 0, 0))
    vnet = jax.vmap(A.net, in_axes=(None, 0, 0))
    reset = lambda n: jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), E.init_state(W))
    push = lambda hist, x: jnp.concatenate([hist[:, 1:], x[:, None]], 1)

    def train_iteration(ag, buf, w, key, ep, cap, pid, trs):
        """One episode on each of the n_envs parallel traces, with G SAC updates every K env steps."""
        n = trs["img"].shape[0]
        steps_per_chunk, n_chunks = args.chunk, T // args.chunk

        def chunk(c, _):
            ag, s, hist, hv, buf, w, key = c

            def env_step(c, _):
                s, hist, hv, buf, w, key = c
                key, ka = jax.random.split(key)
                v = vview(W, cap, pid, s, trs)
                obs, mask = vobs(W, cap, s, v), jax.vmap(E.mask_of)(v)
                hist, hv = push(hist, obs), push(hv, jnp.ones(n, bool))
                a = jax.random.categorical(ka, jnp.where(mask, vnet(ag["actor"], hist, hv), -1e9))
                s2, r, info = vstep(W, pid, s, v, a)
                rvec = jax.vmap(E.reward_vector)(v)
                buf = A.buffer_write(buf, w, obs, mask, a, rvec, s.k == T - 1, jnp.full(n, ep))
                out = {k: info[k] for k in ("latency", "met", "hit", "rejected", "wait")}
                return (s2, hist, hv, buf, w + 1, key), dict(out, reward=r)
            (s, hist, hv, buf, w, key), out = jax.lax.scan(env_step, (s, hist, hv, buf, w, key), None, steps_per_chunk)

            def upd(c2, _):
                ag, key = c2
                key, kb = jax.random.split(key)
                ag2, st = A.sac_update(ag, A.buffer_sample(buf, w, kb, args.batch, H), hp)
                return (ag2, key), st
            def learn(ag, key):
                return jax.lax.scan(upd, (ag, key), None, args.updates)
            def skip(ag, key):
                return (ag, key), jax.tree.map(lambda x: jnp.full((args.updates,), jnp.nan), stats_shape)
            (ag, key), st = jax.lax.cond(w >= args.learn_start, learn, skip, ag, key)
            return (ag, s, hist, hv, buf, w, key), (out, st)

        stats_shape = dict(critic_loss=0., actor_loss=0., alpha=0., entropy=0., q=0.)
        s = reset(n)
        hist, hv = jnp.zeros((n, H, N, E.F)), jnp.zeros((n, H), bool)
        (ag, s, _, _, buf, w, key), (out, st) = jax.lax.scan(chunk, (ag, s, hist, hv, buf, w, key), None, n_chunks)
        m = dict(avg_latency=out["latency"].mean(), completion=out["met"].mean(), hit_rate=out["hit"].mean(),
                 rejection=out["rejected"].mean(), avg_wait=out["wait"].mean(), reward=out["reward"].mean(),
                 evictions=s.evictions.mean().astype(jnp.float32))
        m.update({k: jnp.nanmean(v) for k, v in st.items()})
        return ag, buf, key, m

    def episode(cap, pid, tr, act, key):
        """Deterministic evaluation of one trace. act(v, obs, hist, hv, key) -> action."""
        def body(c, k):
            s, hist, hv = c
            v = E.view(W, cap, pid, s, tr)
            obs = E.observe(W, cap, s, v)
            hist = jnp.concatenate([hist[1:], obs[None]])
            hv = jnp.concatenate([hv[1:], jnp.ones(1, bool)])
            a = act(v, hist, hv, k)
            s2, r, info = E.step(W, pid, s, v, a)
            return (s2, hist, hv), info
        (s, _, _), infos = jax.lax.scan(body, (E.init_state(W), jnp.zeros((H, N, E.F)), jnp.zeros(H, bool)),
                                        jax.random.split(key, T))
        return E.summarize(infos, s, tr["allow"]), infos

    def agent_act(actor):
        def act(v, hist, hv, k):
            return jnp.argmax(jnp.where(E.mask_of(v), A.net(actor, hist, hv), -jnp.inf))
        return act

    def heur_act(name):
        h = E.make_heuristic(W, name)
        return lambda v, hist, hv, k: h(v, k)

    # [C configs] x [V traces]
    eval_agents = jax.jit(jax.vmap(jax.vmap(lambda actor, cap, pid, tr, key: episode(cap, pid, tr, agent_act(actor), key)[0],
                                            in_axes=(None, None, None, 0, None)), in_axes=(0, 0, 0, None, None)))
    eval_heur = {name: jax.jit(jax.vmap(jax.vmap(lambda cap, pid, tr, key, name=name: episode(cap, pid, tr, heur_act(name), key)[0],
                                                 in_axes=(None, None, 0, None)), in_axes=(0, 0, None, None)))
                 for name in E.HEURISTICS}
    record_agent = jax.jit(lambda actor, cap, pid, tr, key: episode(cap, pid, tr, agent_act(actor), key)[1])
    record_heur = {name: jax.jit(lambda cap, pid, tr, key, name=name: episode(cap, pid, tr, heur_act(name), key)[1])
                   for name in E.HEURISTICS}
    it = jax.jit(jax.vmap(train_iteration, in_axes=(0, 0, None, 0, None, 0, 0, None)), donate_argnums=(0, 1))
    return dict(iteration=it, eval_agents=eval_agents, eval_heur=eval_heur, record_agent=record_agent,
                record_heur=record_heur, hp=hp, N=N, I=I)


def configs_of(args):
    from data import POLICIES, DISKS
    return [(p, d, s) for s in args.seeds for p in POLICIES for d in DISKS]


def caps_pids(world, configs):
    import env as E
    from data import POLICIES, DISKS
    return (np.array([E.capacity(world, DISKS[d]) for _, d, _ in configs], np.float32),
            np.array([POLICIES.index(p) for p, _, _ in configs], np.int32))


def as_rows(metrics, configs, extra):
    rows = []
    for c, (p, d, s) in enumerate(configs):
        rows.append(dict(policy=p, disk=d, seed=s, **extra, **{k: float(np.asarray(v)[c]) for k, v in metrics.items()}))
    return rows


# ====================================================================== train
def train(args):
    jax = setup(args.device)
    import jax.numpy as jnp
    import agent as A
    import env as E
    from data import load
    world, train_tr, meta = load(args.data, "train")
    _, val_tr, _ = load(args.data, "val")
    T = train_tr["img"].shape[1]
    assert T % args.chunk == 0, "tasks per trace must be divisible by --chunk"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    configs = configs_of(args)
    C = len(configs)
    caps, pids = caps_pids(world, configs)
    fn = build(world, T, args)
    N, I = fn["N"], fn["I"]
    keys = jax.random.split(jax.random.PRNGKey(args.seed), C + 1)
    init_keys = jnp.stack([jax.random.fold_in(jax.random.PRNGKey(1000 + s), i) for i, (_, _, s) in enumerate(configs)])
    agents = jax.vmap(lambda k: A.init_agent(k, E.F, fn["hp"]))(init_keys)
    agents = jax.tree.map(lambda x: jnp.array(np.asarray(x)), agents)   # distinct buffers (targets start == critics)
    L = args.buffer // args.envs
    buffers = jax.vmap(lambda _: A.init_buffer(args.envs, L, N, E.F, N + 1))(jnp.arange(C))
    run_keys = keys[1:]
    val = {k: jnp.asarray(v) for k, v in val_tr.items()}
    caps_j, pids_j = jnp.asarray(caps), jnp.asarray(pids)
    iters = max(1, args.episodes // args.envs)
    (out / "config.json").write_text(json.dumps(dict(args=vars(args), hp=fn["hp"], configs=configs, data_meta=meta,
                                                     devices=[str(d) for d in jax.devices()], jax=jax.__version__), indent=1))
    print(f"Device: {jax.devices()} | {C} agents (policies x disks x seeds) | {args.envs} parallel envs each | "
          f"{T} tasks/episode | {iters} iterations = {iters * args.envs} episodes per agent", flush=True)

    # Heuristic references on validation traces (once).
    t0 = time.time()
    vkey = jax.random.PRNGKey(7)
    base = {}
    for name in E.HEURISTICS:
        m = jax.device_get(fn["eval_heur"][name](caps_j, pids_j, val, vkey))
        base[name] = {k: v.mean(1) for k, v in m.items()}
        append_csv(out / "validation_baselines.csv", as_rows(base[name], configs, dict(method=name)))
    print(f"Validation baselines computed in {time.time() - t0:.1f}s", flush=True)

    def validate(it_done):
        m = jax.device_get(fn["eval_agents"](agents["actor"], caps_j, pids_j, val, vkey))
        m = {k: v.mean(1) for k, v in m.items()}
        append_csv(out / "validation.csv", as_rows(m, configs, dict(episode=it_done * args.envs)))
        return m

    best_score = np.full(C, -np.inf)
    best_actor = jax.device_get(agents["actor"])

    def keep_best(m):
        nonlocal best_actor
        better = m["reward"] > best_score
        best_score[better] = m["reward"][better]
        cur = jax.device_get(agents["actor"])
        best_actor = jax.tree.map(lambda b, c: np.where(better.reshape((-1,) + (1,) * (c.ndim - 1)), c, b), best_actor, cur)

    m0 = validate(0)
    keep_best(m0)
    rng = np.random.default_rng(args.seed)
    n_train = train_tr["img"].shape[0]
    start, w = time.time(), 0
    log = open(out / "progress.log", "a")
    for it in range(iters):
        idx = rng.choice(n_train, args.envs, replace=n_train < args.envs)
        trs = {k: jnp.asarray(v[idx]) for k, v in train_tr.items()}
        t_it = time.time()
        agents, buffers, run_keys, m = fn["iteration"](agents, buffers, jnp.int32(w), run_keys, jnp.int32(it), caps_j, pids_j, trs)
        m = jax.device_get(m)
        w += T
        done = it + 1
        append_csv(out / "training.csv", as_rows(m, configs, dict(episode=done * args.envs, iteration=done)))
        el = time.time() - start
        eta = el / done * (iters - done) if done > 1 else float("nan")
        line = (f"[iter {done:4d}/{iters}] episodes {done * args.envs:5d}/{iters * args.envs} | {time.time() - t_it:5.1f}s/it | "
                f"elapsed {fmt_time(el)} | ETA {fmt_time(eta) if done > 1 else '...'} | train: lat {np.mean(m['avg_latency']):.2f} "
                f"on-time {np.mean(m['completion']):.3f} hit {np.mean(m['hit_rate']):.3f} | critic {np.nanmean(m['critic_loss']):.3f} "
                f"alpha {np.nanmean(m['alpha']):.4f} ent {np.nanmean(m['entropy']):.2f}")
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
        if done % args.eval_every == 0 or done == iters:
            mv = validate(done)
            keep_best(mv)
            g = base["Greedy_latency"]
            print(f"  validation @ {done * args.envs} episodes (agent vs Greedy_latency, avg latency | on-time):", flush=True)
            for c, (p, d, s) in enumerate(configs):
                print(f"    {p:4s} {d:6s} s{s}: {mv['avg_latency'][c]:6.3f} vs {g['avg_latency'][c]:6.3f} | "
                      f"{mv['completion'][c]:.3f} vs {g['completion'][c]:.3f}{'  *' if mv['avg_latency'][c] < g['avg_latency'][c] else ''}",
                      flush=True)
            with open(out / "models.pkl", "wb") as f:
                pickle.dump(dict(configs=configs, hp=fn["hp"], best_actor=best_actor, best_val_reward=best_score,
                                 last=jax.device_get(agents), episodes=done * args.envs), f)
    log.close()
    print(f"Training done in {fmt_time(time.time() - start)}; checkpoints: {out / 'models.pkl'}", flush=True)


# ====================================================================== evaluate
def evaluate(args):
    jax = setup(args.device)
    import jax.numpy as jnp
    import env as E
    from data import load
    world, test_tr, meta = load(args.data, "test")
    T = test_tr["img"].shape[1]
    out = Path(args.out)
    ck = pickle.loads((out / "models.pkl").read_bytes())
    configs = [tuple(c) for c in ck["configs"]]
    hp = ck["hp"]
    args.width, args.history, args.batch, args.gamma = hp["width"], hp["history"], hp["batch"], hp["gamma"]
    args.ent_ratio = hp.get("ent_ratio", 0.1)
    args.plain_critic = not hp.get("decompose", True)
    fn = build(world, T, args)
    caps, pids = caps_pids(world, configs)
    caps_j, pids_j = jnp.asarray(caps), jnp.asarray(pids)
    test = {k: jnp.asarray(v) for k, v in test_tr.items()}
    key = jax.random.PRNGKey(11)
    actor = jax.tree.map(jnp.asarray, ck["best_actor"])
    rows = []
    t0 = time.time()
    results = {"Agent": jax.device_get(fn["eval_agents"](actor, caps_j, pids_j, test, key))}
    for name in E.HEURISTICS:
        results[name] = jax.device_get(fn["eval_heur"][name](caps_j, pids_j, test, key))
    for method, m in results.items():
        for c, (p, d, s) in enumerate(configs):
            if method != "Agent" and s != configs[0][2]:
                continue                                   # heuristics are seed-independent
            for ti in range(test_tr["img"].shape[0]):
                rows.append(dict(method=method, policy=p, disk=d, seed=s, trace=ti, **{k: float(v[c, ti]) for k, v in m.items()}))
    (out / "results.csv").unlink(missing_ok=True)
    append_csv(out / "results.csv", rows)
    print(f"Test evaluation ({test_tr['img'].shape[0]} traces x {len(configs)} configs x 6 methods) in {time.time() - t0:.1f}s", flush=True)
    # Replay logs on test trace 0 (first --replay_slots slots).
    tr0 = {k: v[0] for k, v in test.items()}
    rdir = out / "replays"
    seed0 = configs[0][2]
    for c, (p, d, s) in enumerate(configs):
        if s != seed0:
            continue
        meta_r = dict(policy=p, disk=d, disk_frac=float(caps[c] / world["img_size"].sum()), capacity=float(caps[c]),
                      trace="test_0000", slot_unit="slot")
        infos = jax.device_get(fn["record_agent"](jax.tree.map(lambda x: x[c], actor), caps_j[c], pids_j[c], tr0, key))
        E.write_replay(rdir / f"Agent_{p}_{d}_test0.json", E.replay_json(world, tr0, infos, dict(method="Agent", **meta_r), args.replay_slots))
        for name in E.HEURISTICS:
            infos = jax.device_get(fn["record_heur"][name](caps_j[c], pids_j[c], tr0, key))
            E.write_replay(rdir / f"{name}_{p}_{d}_test0.json", E.replay_json(world, tr0, infos, dict(method=name, **meta_r), args.replay_slots))
    print(f"Replay logs written to {rdir}", flush=True)
    summary = {}
    for r in rows:
        k = f"{r['method']}|{r['policy']}|{r['disk']}"
        summary.setdefault(k, []).append(r)
    agg = {k: {m: float(np.mean([r[m] for r in v])) for m in ("avg_latency", "completion", "hit_rate", "evictions", "avg_wait", "rejection")}
           for k, v in summary.items()}
    (out / "summary.json").write_text(json.dumps(agg, indent=1))
    print(f"{'config':18s} {'Agent lat':>10s} {'Greedy lat':>10s} {'Agent on-time':>14s} {'Greedy on-time':>15s} {'Agent ev':>9s} {'Greedy ev':>9s}")
    from data import POLICIES, DISKS
    for p in POLICIES:
        for d in DISKS:
            a, g = agg[f"Agent|{p}|{d}"], agg[f"Greedy_latency|{p}|{d}"]
            print(f"{p + '/' + d:18s} {a['avg_latency']:10.3f} {g['avg_latency']:10.3f} {a['completion']:14.3f} {g['completion']:15.3f} "
                  f"{a['evictions']:9.0f} {g['evictions']:9.0f}")


# ====================================================================== smoke
def smoke(args):
    from data import make_dataset, CFG
    root = Path(args.out)
    args.data = str(root / "data")
    make_dataset(args.data, counts=(4, 2, 2), cfg=dict(CFG, tasks=200))
    args.episodes, args.envs, args.chunk, args.updates, args.eval_every = 6, 2, 20, 2, 1
    args.width, args.batch, args.buffer, args.learn_start, args.replay_slots = 16, 32, 800, 40, 25
    train(args)
    evaluate(args)
    import plots
    plots.plot_all(root)
    import viz
    viz.render_scenarios(root, fps=6, slots=8)
    print(f"SMOKE OK (mechanics only, not results): {root.resolve()}", flush=True)


def main():
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("command", choices=("data", "train", "evaluate", "plot", "smoke"))
    pa.add_argument("--data", default="data")
    pa.add_argument("--out", default="outputs/run")
    pa.add_argument("--device", default="auto", choices=("auto", "cpu", "gpu"))
    pa.add_argument("--counts", type=int, nargs=3, default=(256, 8, 16), help="train/val/test traces")
    pa.add_argument("--seeds", type=int, nargs="+", default=[0])
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--episodes", type=int, default=2000, help="training episodes (traces) per agent")
    pa.add_argument("--envs", type=int, default=8, help="parallel environments per agent")
    pa.add_argument("--chunk", type=int, default=50, help="env steps between update bursts")
    pa.add_argument("--updates", type=int, default=8, help="SAC updates per chunk")
    pa.add_argument("--learn_start", type=int, default=1000)
    pa.add_argument("--buffer", type=int, default=32000, help="replay capacity (transitions) per agent")
    pa.add_argument("--width", type=int, default=64)
    pa.add_argument("--history", type=int, default=4)
    pa.add_argument("--batch", type=int, default=256)
    pa.add_argument("--gamma", type=float, default=0.9, help="paper: 0.98 (drifts on large disks, see README)")
    pa.add_argument("--plain_critic", action="store_true", help="ablation: critics regress the full Q (no R + future split)")
    pa.add_argument("--ent_ratio", type=float, default=0.1, help="target entropy = ratio * log(#valid nodes)")
    pa.add_argument("--eval_every", type=int, default=5, help="iterations between validation runs")
    pa.add_argument("--replay_slots", "--replay-slots", type=int, default=200)
    args = pa.parse_args()
    sys.path.insert(0, str(Path(__file__).parent))
    if args.command == "data":
        from data import make_dataset
        make_dataset(args.data, counts=tuple(args.counts))
        print(f"Dataset written to {args.data}")
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "plot":
        import plots
        plots.plot_all(args.out)
    else:
        if args.out == "outputs/run":
            args.out = "outputs/smoke"
        smoke(args)


if __name__ == "__main__":
    main()
