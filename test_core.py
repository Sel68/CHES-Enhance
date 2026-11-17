"""Simulator invariants: capacity, pinning, eviction order, type constraints.  Run: python -m pytest -q test_core.py"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from data import make_world, make_trace, CFG, DISKS, POLICIES
import env as E

CFG_T = dict(CFG, tasks=300)
WORLD = make_world(CFG_T)
W = E.to_jax(WORLD)
TR = {k: jnp.asarray(v) for k, v in make_trace(WORLD, "val", 0, CFG_T).items()}
view = jax.jit(E.view)
step = jax.jit(E.step)


def run(policy, disk, chooser="Greedy_latency", n=300):
    cap, pid = E.capacity(WORLD, DISKS[disk]), POLICIES.index(policy)
    act = jax.jit(E.make_heuristic(W, chooser))
    s, key, log = E.init_state(W), jax.random.PRNGKey(0), []
    for i in range(n):
        v = view(W, cap, pid, s, TR)
        a = act(v, jax.random.fold_in(key, i))
        log.append((s, v, int(a)))
        s, r, info = step(W, pid, s, v, a)
        log[-1] += (info,)
    return cap, pid, s, log


def reference_victims(s, v, n, pid):
    """Sequential greedy eviction written plainly (the JAX version sorts once and takes a prefix)."""
    s = jax.device_get(s)
    now = float(v["now"])
    sizes = WORLD["img_size"]
    on = s.job_fin[n] > now
    pinned = {int(i) for i in s.job_img[n][on]}
    cand = [i for i in range(len(sizes)) if s.present[n, i] and i not in pinned and s.ready[n, i] <= now]
    keyf = {0: lambda i: (s.last[n, i],), 1: lambda i: (s.freq[n, i], s.last[n, i]),
            2: lambda i: (s.ins[n, i], s.last[n, i]), 3: lambda i: (s.key[n, i], s.last[n, i])}[pid]
    need, out = float(v["need"][n]), []
    for i in sorted(cand, key=keyf):
        if need <= 1e-6:
            break
        out.append(i)
        need -= sizes[i]
    return sorted(out)


@pytest.mark.parametrize("policy", POLICIES)
def test_capacity_pins_and_eviction_order(policy):
    cap, pid, s, log = run(policy, "tight")
    sizes = WORLD["img_size"]
    evictions = 0
    for st, v, a, info in log:
        if a < len(WORLD["node_x"]) and bool(info["miss"]):
            got = sorted(np.flatnonzero(np.asarray(info["evicted"])).tolist())
            assert got == reference_victims(st, v, a, pid)
            evictions += len(got)
            now = float(v["now"])
            on = np.asarray(st.job_fin[a]) > now
            assert not set(got) & set(np.asarray(st.job_img[a])[on].tolist()), "evicted a pinned image"
    for st, *_ in log:
        used = (np.asarray(st.present) * sizes).sum(1)
        assert (used <= cap + 1e-3).all(), "disk capacity exceeded"
    assert evictions > 0, "tight disk should evict"


def test_large_disk_never_evicts():
    _, _, s, _ = run("LRU", "large")
    assert int(s.evictions) == 0


def test_type_constraint_and_masks():
    _, _, _, log = run("GDSF", "medium", chooser="Random")
    gpu = WORLD["node_gpu"]
    for st, v, a, info in log:
        if a < len(gpu):
            assert bool(v["feas"][a])
            assert not (gpu[a] and not WORLD["img_gpu"][int(v["img"])]), "CPU service placed on a GPU node"


def test_latency_matches_estimate():
    """The scheduler's latency estimate equals the realised latency (deterministic model)."""
    _, _, _, log = run("LRU", "tight")
    for st, v, a, info in log[:100]:
        if a < len(WORLD["node_x"]):
            assert abs(float(info["latency"]) - float(v["lat"][a])) < 1e-4
