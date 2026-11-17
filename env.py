"""JAX-native CHES simulator: GPU-CPU heterogeneous nodes, serial image pulls, finite disks,
LRU / LFU / FIFO / GDSF eviction. Pure functions -> jit / vmap over configs and parallel envs.

Latency model (paper Eq. 1-6, authors' code units = slots):
  T = max(T_upload, T_image_wait) + T_comp
  T_image_wait : 0 if cached, residual pull if downloading, else node pull-queue + size/(2*bw)
  T_comp (CPU node) = c * U / F                 (GPU service on a CPU node needs 4.5x the work)
  T_comp (GPU node) = a*d*U/Bg + f*U / (Fg * (1-lambda)^(U-1))
U = concurrent jobs on the node at admission (incl. the new one). Jobs hold memory and pin
their image from admission until finish. Reward = (deadline allowance - latency) / R_SCALE (Eq. 18).
"""
from typing import NamedTuple
import json
import jax
import jax.numpy as jnp
import numpy as np
from data import POLICIES, CFG

F = 29            # per-node observation features
PEN = 5.0         # rejected task: latency = allowance + PEN slots
R_SCALE = 5.0
HEURISTICS = ("Greedy_latency", "Image_priority", "Least_load", "Greedy_resources", "Random")


class State(NamedTuple):
    k: jnp.ndarray          # index of the current task
    present: jnp.ndarray    # [N,I] image resident (ready or downloading)
    ready: jnp.ndarray      # [N,I] time the pull finishes
    start: jnp.ndarray      # [N,I] time the pull started
    last: jnp.ndarray       # [N,I] last access (task counter)
    ins: jnp.ndarray        # [N,I] insertion (task counter)
    freq: jnp.ndarray       # [N,I] resident reference count
    key: jnp.ndarray        # [N,I] GDSF priority H
    clock: jnp.ndarray      # [N]   GDSF inflation value L
    dl_end: jnp.ndarray     # [N]   end of the node's serial pull queue
    job_fin: jnp.ndarray    # [N,J] finish time (-inf = empty)
    job_img: jnp.ndarray    # [N,J]
    job_mem: jnp.ndarray    # [N,J]
    demand: jnp.ndarray     # [I]   causal EMA of requested images
    evictions: jnp.ndarray
    downloads: jnp.ndarray
    dl_gb: jnp.ndarray


def to_jax(world):
    return {k: jnp.asarray(v) for k, v in world.items()}


def capacity(world, frac):
    sizes = np.asarray(world["img_size"])
    return float(max(sizes.max(), frac * sizes.sum()))


def init_state(world, J=CFG["max_jobs"]):
    N, I = world["node_gpu"].shape[0], world["img_size"].shape[0]
    z = jnp.zeros((N, I), jnp.float32)
    i0 = jnp.array(0, jnp.int32)
    return State(i0, jnp.zeros((N, I), bool), z, z, z, z, z, z, jnp.zeros(N), jnp.zeros(N),
                 jnp.full((N, J), -jnp.inf), jnp.zeros((N, J), jnp.int32), jnp.zeros((N, J)),
                 jnp.zeros(I), i0, i0, jnp.array(0., jnp.float32))


def task_at(tr, k):
    return {name: v[k] for name, v in tr.items()}


def view(W, cap, pid, s, tr):
    """Everything the scheduler can see for the current task (and the eviction plan per node)."""
    t = task_at(tr, s.k)
    N, I = s.present.shape
    J = s.job_fin.shape[1]
    now = t["slot"].astype(jnp.float32)
    img, sizes = t["img"], W["img_size"]
    size, tgpu = sizes[img], W["img_gpu"][img]
    on = s.job_fin > now
    U = on.sum(1)
    mem_used = (s.job_mem * on).sum(1)
    pins = (on[:, :, None] & (s.job_img[:, :, None] == jnp.arange(I))).sum(1)
    used = (s.present * sizes).sum(1)
    busy = s.present & ((pins > 0) | (s.ready > now))
    protected = (busy * sizes).sum(1)
    has = s.present[:, img]
    ready_i = s.ready[:, img]
    hit = has & (ready_i <= now)
    need = jnp.where(has, 0., jnp.maximum(0., used + size - cap))
    # Eviction plan: evict candidates in policy order until `need` GB are free (exact greedy eviction).
    cand = s.present & ~busy
    primary = jnp.stack([s.last, s.freq, s.ins, s.key])[pid]          # LRU, LFU, FIFO, GDSF
    primary = jnp.where(cand, primary, jnp.inf)
    order = jnp.lexsort((s.last, primary), axis=-1)
    sz = jnp.take_along_axis(sizes[None] * cand, order, -1)
    ev_sorted = jnp.take_along_axis(cand, order, -1) & ((jnp.cumsum(sz, -1) - sz) < need[:, None])
    evict = jnp.take_along_axis(ev_sorted, jnp.argsort(order, -1), -1)
    # Feasibility (paper Alg. 2): type, memory, concurrency slot, disk.
    feas = (~(W["node_gpu"] & ~tgpu)) & (mem_used + t["mem"] <= W["node_mem"]) & (U < J) & \
           (has | (protected + size <= cap))
    # Latency estimate == realised latency in this model (deterministic given state and action).
    dl_time = size / (CFG["dl_div"] * W["node_bw"])
    wait = jnp.where(has, jnp.maximum(ready_i - now, 0.), jnp.maximum(s.dl_end - now, 0.) + dl_time)
    U1 = (U + 1).astype(jnp.float32)
    comp_cpu = jnp.where(tgpu, 4.5, 1.) * t["work"] * U1 / W["node_speed"]
    comp_gpu = CFG["gpu_alpha"] * t["data"] * U1 / W["node_gbw"] + \
        t["work"] * U1 / (W["node_speed"] * (1 - CFG["gpu_lambda"]) ** (U1 - 1))
    comp = jnp.where(W["node_gpu"], comp_gpu, comp_cpu)
    dist = jnp.hypot(t["x"] - W["node_x"], t["y"] - W["node_y"])
    se = jnp.log2(1 + 100 / (1 + (dist / 25) ** 2)) / jnp.log2(101.)
    upload = t["data"] / (W["node_bw"] * se)
    lat = jnp.maximum(upload, wait) + comp
    return dict(t=t, now=now, img=img, size=size, tgpu=tgpu, U=U, mem_used=mem_used, used=used, has=has,
                hit=hit, ready_i=ready_i, need=need, evict=evict, feas=feas, dl_time=dl_time, wait=wait,
                comp=comp, upload=upload, lat=lat)


def reward_vector(v):
    """Immediate reward (Eq. 18) of every action: nodes, then forced rejection. Known before acting."""
    return jnp.concatenate([(v["t"]["allow"] - v["lat"]) / R_SCALE, jnp.array([-PEN / R_SCALE])])


def mask_of(v):
    return jnp.concatenate([v["feas"], ~v["feas"].any()[None]])   # forced rejection only if nothing fits


def observe(W, cap, s, v):
    t, now = v["t"], v["now"]
    N = s.present.shape[0]
    total = W["img_size"].sum()
    allow = t["allow"]
    best = jnp.min(jnp.where(v["feas"], v["lat"], jnp.inf))
    best = jnp.where(jnp.isfinite(best), best, 0.)
    victim_value = (s.demand * v["evict"]).sum(1)
    b = lambda x: jnp.broadcast_to(x, (N,))
    cols = [W["node_gpu"], W["node_speed"] / 16, W["node_bw"] / 6, b(cap / total), W["node_x"] / 100, W["node_y"] / 100,
            v["U"] / s.job_fin.shape[1], v["mem_used"] / W["node_mem"], v["used"] / cap,
            jnp.clip(s.dl_end - now, 0, 12) / 4, (s.demand * s.present).sum(1) / (s.demand.sum() + 1e-3),
            v["hit"], v["has"] & ~v["hit"], jnp.clip(v["wait"], 0, 12) / 4, jnp.clip(v["comp"], 0, 12) / 4,
            jnp.clip(v["upload"], 0, 4), jnp.clip(v["lat"] / allow, 0, 4), jnp.clip((v["lat"] - best) / allow, 0, 4),
            v["need"] / v["size"], jnp.clip(victim_value * 5, 0, 4), v["evict"].sum(1) / 3, v["feas"],
            b(v["tgpu"]), b(t["work"] / 5), b(t["mem"] / 10), b(allow / 6), b(v["size"] / 15),
            b(s.demand[v["img"]] * 5), b(v["has"].mean())]
    return jnp.stack([c.astype(jnp.float32) for c in cols], -1)


def step(W, pid, s, v, action):
    """Apply an action (node index, or N = reject). Returns new state, reward, info."""
    N, I = s.present.shape
    t, now, img = v["t"], v["now"], v["img"]
    rej = action >= N
    n = jnp.minimum(action, N - 1)
    node = (jnp.arange(N) == n) & ~rej
    cell = node[:, None] & (jnp.arange(I) == img)[None]
    has, ready_old = v["has"][n], v["ready_i"][n]
    miss = ~has & ~rej
    ev = v["evict"][n] & miss
    clock_n = jnp.where(pid == 3, jnp.maximum(s.clock[n], jnp.max(jnp.where(ev, s.key[n], -jnp.inf))), s.clock[n])
    present = s.present & ~(node[:, None] & ev[None])
    dl_start = jnp.maximum(now, s.dl_end[n])
    dl_ready = dl_start + v["dl_time"][n]
    mc = cell & miss
    k = s.k.astype(jnp.float32)
    freq = jnp.where(mc, 0., s.freq) + cell
    cost = 1 + v["dl_time"][n]
    key = jnp.where(cell, clock_n + freq * cost / v["size"], s.key)
    ready_img = jnp.where(miss, dl_ready, ready_old)
    fin = now + jnp.maximum(v["upload"][n], jnp.maximum(ready_img - now, 0.)) + v["comp"][n]
    lat = jnp.where(rej, t["allow"] + PEN, fin - now)
    slot = jnp.argmin(s.job_fin[n] > now)                 # first free job slot (feasibility guarantees one)
    jn = node[:, None] & (jnp.arange(s.job_fin.shape[1]) == slot)[None]
    s2 = State(s.k + 1, present | mc, jnp.where(mc, dl_ready, s.ready), jnp.where(mc, dl_start, s.start),
               jnp.where(cell, k, s.last), jnp.where(mc, k, s.ins), freq, key,
               jnp.where(node, clock_n, s.clock), jnp.where(node & miss, dl_ready, s.dl_end),
               jnp.where(jn, fin, s.job_fin), jnp.where(jn, img, s.job_img), jnp.where(jn, t["mem"], s.job_mem),
               0.98 * s.demand + 0.02 * (jnp.arange(I) == img), s.evictions + ev.sum(),
               s.downloads + miss, s.dl_gb + miss * v["size"])
    met = ~rej & (lat <= t["allow"])
    reward = (t["allow"] - lat) / R_SCALE
    info = dict(node=jnp.where(rej, -1, n), hit=v["hit"][n] & ~rej, coalesced=has & ~v["hit"][n] & ~rej, miss=miss,
                rejected=rej, evicted=ev, dl_start=jnp.where(rej | v["hit"][n], jnp.nan, jnp.where(miss, dl_start, s.start[n, img])),
                dl_ready=jnp.where(rej | v["hit"][n], jnp.nan, ready_img), upload_fin=now + v["upload"][n],
                finish=jnp.where(rej, jnp.nan, fin), latency=lat, met=met, wait=jnp.where(rej, 0., jnp.maximum(ready_img - now, 0.)))
    return s2, reward, info


def heuristic(name, v, key):
    """Baselines from the paper; all share masks and physics. Returns N if nothing is feasible."""
    feas, N = v["feas"], v["feas"].shape[0]
    noise = jax.random.uniform(key, (N,))
    inf = jnp.inf
    if name == "Greedy_latency":
        score = v["lat"]
    elif name == "Image_priority":                       # cached image first, else random feasible node
        cached = feas & v["has"]
        score = jnp.where(cached.any(), jnp.where(cached, v["lat"], inf), noise)
    elif name == "Least_load":
        score = v["U"] + 0.01 * noise
    elif name == "Greedy_resources":
        score = -v["cap_speed"] / (v["U"] + 1.)
    else:
        score = noise
    a = jnp.argmin(jnp.where(feas, score, inf))
    return jnp.where(feas.any(), a, N)


def make_heuristic(W, name):
    def act(v, key):
        if name == "Greedy_resources":   # highest free compute share x free memory fraction
            v = dict(v, cap_speed=W["node_speed"] * (1 - v["mem_used"] / W["node_mem"]))
        return heuristic(name, v, key)
    return act


def summarize(infos, s, allow):
    """Episode metrics from stacked per-step infos (arrays [T])."""
    lat = infos["latency"]
    return dict(avg_latency=lat.mean(), completion=infos["met"].mean(), hit_rate=infos["hit"].mean(),
                rejection=infos["rejected"].mean(), avg_wait=infos["wait"].mean(),
                reward=((allow - lat) / R_SCALE).mean(), evictions=s.evictions.astype(jnp.float32),
                downloads=s.downloads.astype(jnp.float32), dl_gb=s.dl_gb)


# ---------------------------------------------------------------- replay export (host side, eval only)
def replay_json(world, tr, infos, meta, slots=200):
    """Compact event log for viz.py. `infos` are numpy arrays [T,...] for one trace."""
    tr = {k: np.asarray(v) for k, v in tr.items()}
    infos = {k: np.asarray(v) for k, v in infos.items()}
    keep = np.flatnonzero(tr["slot"] < slots)
    r = lambda x: None if not np.isfinite(x) else round(float(x), 3)
    tasks = []
    for i in keep:
        tasks.append([int(i), int(tr["slot"][i]), int(world["img_gpu"][tr["img"][i]]), int(tr["img"][i]),
                      int(infos["node"][i]), 1 if infos["hit"][i] else (2 if infos["coalesced"][i] else (3 if infos["miss"][i] else 0)),
                      [int(j) for j in np.flatnonzero(infos["evicted"][i])], r(infos["dl_start"][i]), r(infos["dl_ready"][i]),
                      r(infos["finish"][i]), int(infos["met"][i]), round(float(tr["x"][i]), 1), round(float(tr["y"][i]), 1),
                      round(float(tr["slot"][i] + tr["allow"][i]), 2)])
    nodes = [dict(x=round(float(world["node_x"][n]), 1), y=round(float(world["node_y"][n]), 1),
                  gpu=bool(world["node_gpu"][n])) for n in range(len(world["node_x"]))]
    return dict(version=2, **meta, slots=int(min(slots, tr["slot"].max() + 1)), nodes=nodes,
                images=[dict(size=round(float(s), 2), gpu=bool(g)) for s, g in zip(world["img_size"], world["img_gpu"])],
                task_fields=["id", "slot", "gpu_task", "image", "node", "outcome(1hit,2coalesced,3miss,0reject)",
                             "evicted", "dl_start", "dl_ready", "finish", "met", "x", "y", "deadline"],
                tasks=tasks)


def write_replay(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":")))
