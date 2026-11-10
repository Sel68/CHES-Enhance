"""Physical/cache invariants, split independence, SAC gradients and replay contracts."""
import json
import os
import subprocess
import sys
import numpy as np
import pytest
from data import Config, world, trace, POLICIES
from env import Env
from run import rollout


def tiny(tasks=None):
    w = world(Config(cpu_nodes=1, gpu_nodes=1, images=4, slots=2))
    for i in w["images"]:
        i["size"] = 1.
    for n in w["nodes"]:
        n.update(speed=2., memory=100., bandwidth=8.)
    requests = dict(id="test_0000", split="test", index=0, slots=2, tasks=tasks or [task(0), task(1)])
    return w, requests


def task(i, image=0, arrival=0., work=2., kind="CPU"):
    return dict(id=i, slot=int(arrival), arrival=arrival, image=image, type=kind,
                work=work, memory=1., data=0., x=0., y=0., deadline=arrival + 20)


def preload(env, images=(0,)):
    n = env.nodes[0]
    for i in images:
        n["cache"][i] = dict(size=1., ready=0., start=-1., inserted=i, last=i, freq=1, pins=0, key=1.)


def test_processor_sharing_reallocates_at_completion():
    w, tr = tiny([task(0, work=2), task(1, work=4)])
    env = Env(w, tr, disk="large")
    preload(env)
    env.step(0); env.step(0)
    assert env.jobs[0]["finish"] == pytest.approx(2.)
    assert env.jobs[1]["finish"] == pytest.approx(3.)
    assert not env.nodes[0]["active"]
    assert env.nodes[0]["cache"][0]["pins"] == 0


def test_download_coalescing_and_miss_is_not_hit():
    w, tr = tiny()
    env = Env(w, tr, disk="large")
    env.step(0); env.step(0)
    assert env.downloads == 1
    assert not env.jobs[0]["hit"] and not env.jobs[1]["hit"]
    assert env.jobs[1]["coalesced"]
    assert env.jobs[0]["download_finish"] == 1.
    assert env.jobs[0]["finish"] == pytest.approx(3.)


def test_download_queue_is_serial():
    w, tr = tiny([task(0), task(1, image=1)])
    env = Env(w, tr, disk="large")
    env.step(0); env.step(0)
    assert env.jobs[0]["download_finish"] == 1.
    assert env.jobs[1]["download_start"] == 1.
    assert env.jobs[1]["download_finish"] == 2.


def test_pinned_or_downloading_images_cannot_be_evicted():
    w, tr = tiny([task(0), task(1, image=1)])
    env = Env(w, tr, disk="tight")
    env.step(0)
    assert not env.mask()[0]
    assert env.mask()[-1]  # CPU task cannot use GPU node.
    with pytest.raises(ValueError):
        env.step(0)
    env.step(len(env.nodes))
    assert env.jobs[1]["rejected"] and not env.jobs[1]["deadline_met"]
    assert env.evictions == 0


@pytest.mark.parametrize("policy,expected", [("FIFO", 0), ("LRU", 1), ("LFU", 1), ("GDSF", 1)])
def test_victim_semantics(policy, expected):
    w, tr = tiny()
    env = Env(w, tr, policy=policy, disk="large")
    preload(env, (0, 1))
    env.nodes[0]["cache"][0].update(last=10, freq=5, key=5.)
    assert env.victim(env.nodes[0]) == expected


@pytest.mark.parametrize("policy", POLICIES)
def test_real_eviction_after_unpin_and_disk_conservation(policy):
    w, tr = tiny([task(0, work=.1), task(1, image=1, arrival=3.)])
    env = Env(w, tr, policy, "tight", record=True)
    env.step(0); env.step(0)
    assert env.evictions == 1
    assert env.jobs[1]["evicted"] == [dict(node=0, image=0, size=1.)]
    for n in env.nodes:
        assert env.used(n) <= n["capacity"] + 1e-8
        assert all(c["pins"] == 0 for c in n["cache"].values())
    json.dumps(env.replay("Agent", 0), allow_nan=False)


def test_train_val_test_independence_and_reproducibility():
    w = world(Config(slots=10))
    a, b, c = [trace(w, s) for s in ("train", "val", "test")]
    assert a == trace(w, "train")
    assert a["tasks"] != b["tasks"] != c["tasks"]
    assert len({t["image"] for t in a["tasks"]}) < len(a["tasks"])


@pytest.mark.parametrize("policy", POLICIES)
def test_large_disk_is_no_eviction_control(policy):
    w = world(Config(cpu_nodes=3, gpu_nodes=2, images=8, slots=25, max_tasks=6))
    env, _ = rollout(w, trace(w, "test"), policy, "large", method="Greedy_latency")
    assert env.evictions == 0
    assert len(env.results) == len(env.trace["tasks"])
    assert np.isfinite(env.observation()).all()


def test_history_sampling_and_sac_masked_gradients(tmp_path):
    import jax
    from agent import SAC, Replay
    shape = (2, 8)
    rng = np.random.default_rng(1)
    replay = Replay(history=3)
    states = rng.normal(size=(6, *shape)).astype(np.float32)
    masks = np.array([[True, False, False]] * 5 + [[False, False, True]])
    replay.add(states, masks, [0] * 5, [.2] * 5)
    batch = replay.sample(4, rng)
    agent = SAC(shape, width=8, history=3)
    old = jax.tree.map(np.array, agent.critics)
    diagnostics = agent.update(batch)
    assert all(np.isfinite(v) for v in diagnostics.values())
    assert any(np.any(a != b) for a, b in zip(jax.tree.leaves(old), jax.tree.leaves(agent.critics)))
    assert agent.act(np.ones((3, *shape)), np.array([False, True, False]), rng) == 1
    agent.save(tmp_path / "model.pkl", dict(test=True))
    restored = SAC.load(tmp_path / "model.pkl")
    assert restored.metadata["test"]
    for a, b in zip(jax.tree.leaves(agent.actor), jax.tree.leaves(restored.actor)):
        np.testing.assert_array_equal(a, b)


def test_replay_history_does_not_cross_episode_boundary():
    from agent import Replay
    replay = Replay(capacity=5, history=3)
    for val in (1., 9.):
        replay.add(np.full((3, 2, 2), val), np.ones((3, 3), bool), [0, 1], [0, 0])
    batch = replay.sample(100, np.random.default_rng(0))
    for state in batch["s"]:
        assert not (np.any(state == 1) and np.any(state == 9))


def test_actor_and_forced_rejection_critic_learn():
    import jax
    from agent import SAC
    rng = np.random.default_rng(7)
    agent = SAC((2, 8), width=8, history=2)
    batch = dict(s=rng.normal(size=(4, 2, 2, 8)).astype(np.float32),
                 ns=rng.normal(size=(4, 2, 2, 8)).astype(np.float32),
                 m=np.array([[True, True, False]] * 2 + [[False, False, True]] * 2),
                 nm=np.array([[True, True, False]] * 4),
                 a=np.array([0, 1, 2, 2]), r=np.array([.5, .2, -1., -1.]), d=np.zeros(4))
    old_actor = jax.tree.map(np.array, agent.actor)
    old_reject = np.array(agent.critics[0]["reject"])
    agent.update(batch)
    assert np.any(np.array(agent.critics[0]["reject"]) != old_reject)
    assert any(np.any(a != b) for a, b in zip(jax.tree.leaves(old_actor), jax.tree.leaves(agent.actor)))


def test_calibration_has_its_own_seed_domain():
    w = world(Config(slots=4))
    assert trace(w, "calibration")["tasks"] != trace(w, "test")["tasks"]


def test_gdsf_clock_never_reverses_after_a_pinned_entry_is_released():
    w, tr = tiny([task(0, image=1)])
    env = Env(w, tr, "GDSF", "tight")
    preload(env)
    n = env.nodes[0]
    n["clock"] = 10.
    n["cache"][0]["key"] = 1.  # Its key predates inflation while it was pinned.
    env.step(0)
    assert n["clock"] == 10.
    assert n["cache"][1]["key"] > 10.


def test_html_export_has_no_learning_or_plot_dependencies(tmp_path):
    w, tr = tiny()
    env, _ = rollout(w, tr, "LRU", "large", method="Greedy_latency", record=True)
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(env.replay("Greedy_latency", 0)))
    script = "import sys; sys.modules.update({k:None for k in ['jax','torch','numpy','matplotlib','env']}); import viz; viz.export_html(viz.load_panels([sys.argv[1]]),sys.argv[2])"
    subprocess.run([sys.executable, "-c", script, str(path), str(tmp_path / "out.html")], check=True)
    html = (tmp_path / "out.html").read_text()
    assert "__DATA__" not in html and "__PALETTE__" not in html
    assert "<script src=" not in html and 'type="application/json"' in html


def test_video_falls_back_to_real_gif_without_ffmpeg(tmp_path, monkeypatch):
    from matplotlib.animation import writers
    from viz import export_video
    w, tr = tiny()
    env, _ = rollout(w, tr, "LRU", "large", method="Greedy_latency", record=True)
    monkeypatch.setattr(writers, "is_available", lambda _: False)
    path = export_video([env.replay("Greedy_latency", 0)], tmp_path / "fallback.mp4", fps=2, slots=1)
    assert path.suffix == ".gif" and path.read_bytes().startswith(b"GIF")


def test_cli_resume_retains_updates_and_history(tmp_path):
    import csv
    import pickle
    from data import make_dataset
    data_root, output = tmp_path / "data", tmp_path / "output"
    make_dataset(data_root, Config(cpu_nodes=1, gpu_nodes=1, images=4, slots=2,
                                  min_tasks=2, max_tasks=3), (1, 1, 1))
    command = [sys.executable, "run.py", "train", "--data", str(data_root),
               "--output", str(output), "--device", "cpu", "--policies", "LRU",
               "--disks", "tight", "--seeds", "0", "--width", "8", "--history", "2",
               "--batch-size", "2", "--warmup-episodes", "1", "--bc-updates", "0",
               "--eval-every", "1", "--update-ratio", ".01"]
    subprocess.run(command + ["--episodes", "1"], check=True)
    checkpoint = output / "models" / "LRU_tight_s0" / "last.pkl"
    before = pickle.loads(checkpoint.read_bytes())
    subprocess.run(command + ["--episodes", "2", "--resume"], check=True)
    after = pickle.loads(checkpoint.read_bytes())
    subprocess.run(command + ["--episodes", "2", "--resume"], check=True)
    assert before["updates"] == 1 and after["updates"] == 2
    assert after["metadata"]["resumed_from_episode"] == 1
    assert after["metadata"]["replay_rebuilt_on_resume"]
    assert "sampler_state" in after["metadata"]
    with (output / "validation.csv").open() as f:
        assert [int(r["episode"]) for r in csv.DictReader(f)] == [0, 1, 2]
    with (output / "training.csv").open() as f:
        assert [int(r["episode"]) for r in csv.DictReader(f)] == [1, 2]


def test_local_branch_is_learned_and_old_checkpoints_still_load(tmp_path):
    import pickle
    import jax
    from agent import SAC
    rng = np.random.default_rng(5)
    shape = (2, 8)
    plain = SAC(shape, seed=2, width=8, history=2)
    local = SAC(shape, seed=2, width=8, history=2, local_branch=True)
    states = rng.normal(size=(4, 2, *shape)).astype(np.float32)
    masks = np.array([[True, True, False]] * 4)
    np.testing.assert_array_equal(plain.infer(plain.actor, states, masks), local.infer(local.actor, states, masks))
    batch = dict(s=states, ns=states, m=masks, nm=masks, a=np.array([0, 1, 0, 1]),
                 r=np.array([.2, -.3, .4, -.2]), d=np.ones(4))
    local.update(batch)
    assert np.any(np.asarray(local.actor["local"]) != 0)
    local.save(tmp_path / "local.pkl")
    restored = SAC.load(tmp_path / "local.pkl")
    assert "local" in restored.actor
    np.testing.assert_array_equal(local.infer(local.actor, states, masks), restored.infer(restored.actor, states, masks))
    plain.save(tmp_path / "legacy.pkl")
    payload = pickle.loads((tmp_path / "legacy.pkl").read_bytes())
    payload.pop("local_branch")  # Original checkpoint format had no architecture flag.
    (tmp_path / "legacy.pkl").write_bytes(pickle.dumps(payload))
    legacy = SAC.load(tmp_path / "legacy.pkl")
    assert "local" not in legacy.actor
    np.testing.assert_array_equal(plain.infer(plain.actor, states, masks), legacy.infer(legacy.actor, states, masks))


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("disk", ["tight", "medium", "large"])
def test_causal_rewards_equal_episode_slack_and_integrated_latency(policy, disk):
    w = world(Config(cpu_nodes=2, gpu_nodes=1, images=6, slots=8, max_tasks=5))
    env, experience = rollout(w, trace(w, "train"), policy, disk, method="Greedy_latency", collect=True, reward_mode="holding")
    latency = sum(e["latency"] for e in env.jobs.values())
    slack = sum((e["deadline"] - e["finish"]) / 50 for e in env.jobs.values())
    assert env.latency_cost == pytest.approx(latency, abs=1e-6)
    assert sum(experience[3]) == pytest.approx(slack, abs=1e-6)


def test_admission_reward_is_causal_despite_future_contention():
    w, tr = tiny()
    w["nodes"].append(dict(w["nodes"][0], id=2))
    rewards, finishes = [], []
    for second_node in [0, 2]:
        env = Env(w, tr, disk="large")
        previous = env.latency_cost
        env.step(0)
        rewards.append((20 - (env.latency_cost - previous)) / 50)
        env.step(second_node)
        finishes.append(env.jobs[0]["finish"])
    assert rewards[0] == rewards[1] == .4
    assert finishes[0] != finishes[1]


def test_horizon_feature_and_legacy_observation_shape():
    w, tr = tiny([task(0), task(1, arrival=1.)])
    env = Env(w, tr)
    assert env.observation()[0, 18] == 1.
    env.step(0)
    assert env.observation()[0, 18] == 0.
    old_world = dict(w)
    old_world.pop("observation_version")
    legacy = Env(old_world, tr)
    assert legacy.obs_shape[-1] + 1 == env.obs_shape[-1]


def test_state_potential_does_not_modify_simulation_and_matches_real_drain():
    from copy import deepcopy
    w, tr = tiny([task(0, work=3), task(1, image=1, kind="GPU"), task(2, image=2)])
    env = Env(w, tr, disk="large")
    env.step(0); env.step(1)
    before = deepcopy(env.nodes)
    predicted = env.remaining_cost()
    assert env.nodes == before
    incurred = env.latency_cost
    env.advance(env.time + 1000)
    assert predicted == pytest.approx(env.latency_cost - incurred, abs=1e-7)


def test_potential_shaping_preserves_discounted_return_on_same_trajectory():
    w = world(Config(cpu_nodes=2, gpu_nodes=1, images=6, slots=8, max_tasks=5))
    request = trace(w, "train")
    base_env, base = rollout(w, request, "LRU", "tight", method="Greedy_latency", collect=True, reward_mode="holding")
    shaped_env, shaped = rollout(w, request, "LRU", "tight", method="Greedy_latency", collect=True, reward_mode="potential")
    discounts = .98 ** np.arange(len(base[3]))
    assert np.dot(discounts, base[3]) == pytest.approx(np.dot(discounts, shaped[3]), abs=1e-7)
    assert base_env.metrics() == shaped_env.metrics()


def test_evaluation_never_uses_the_training_potential(monkeypatch):
    w, request = tiny()
    def forbidden(_):
        raise AssertionError("Training projection used during evaluation")
    monkeypatch.setattr(Env, "remaining_cost", forbidden)
    class Actor:
        history = 1
        def act(self, history, mask, rng):
            return int(np.flatnonzero(mask)[0])
    env, experience = rollout(w, request, "LRU", "large", agent=Actor(), record=True)
    assert experience is None and len(env.results) == 2
