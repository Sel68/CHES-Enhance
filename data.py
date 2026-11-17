"""World (nodes + image/service catalogue) and request traces, stored as compact .npz arrays.

Units follow the authors' code: time is measured in slots, image download takes
size_GB / bandwidth_Gbps slots (authors divide by 2; we use the slower pull to make cold starts matter), deadlines are a few slots after arrival.
Services recur (Zipf popularity with a drifting hot set + short-term repeats), so
an image evicted now may be needed again soon: this is where disk size and the
eviction policy matter.
"""
from pathlib import Path
import json
import numpy as np

POLICIES = ("LRU", "LFU", "FIFO", "GDSF")
# Per-node disk capacity as a fraction of the whole catalogue (~450 GB).
# tight ~ 2-3 images/node (heavy eviction), medium ~ 5 images/node, large = everything (no eviction).
DISKS = {"tight": 0.05, "medium": 0.10, "large": 1.0}
SPLIT_SEEDS = {"train": 101, "val": 202, "test": 303}

CFG = dict(seed=2026, cpu_nodes=14, gpu_nodes=6, images=50, tasks=2000, max_jobs=16,
           gpu_lambda=0.05, gpu_alpha=0.5, phase_slots=40, zipf=1.0, repeat=0.2,
           rate=14.0, allow=(2.5, 5.0), dl_div=1.0)


def make_world(cfg=CFG):
    rng = np.random.default_rng(cfg["seed"])
    n_c, n_g, n_i = cfg["cpu_nodes"], cfg["gpu_nodes"], cfg["images"]
    gpu = np.r_[np.zeros(n_c), np.ones(n_g)].astype(bool)
    x, y = rng.uniform(5, 95, n_c + n_g), rng.uniform(5, 95, n_c + n_g)
    order = np.lexsort((x, gpu))              # CPU nodes then GPU nodes, each sorted by x (CNN locality)
    gpu, x, y = gpu[order], x[order], y[order]
    n = n_c + n_g
    world = dict(
        node_gpu=gpu, node_x=x, node_y=y,
        node_speed=np.where(gpu, rng.uniform(10, 16, n), rng.uniform(3.0, 6.5, n)),   # GPU FLOPS / CPU GHz
        node_mem=np.where(gpu, rng.uniform(16, 32, n), rng.uniform(80, 180, n)),     # GPU video mem / RAM (GB)
        node_bw=rng.uniform(2, 6, n),                                                 # Gbps
        node_gbw=rng.uniform(2, 6, n),                                                # GPU memory bandwidth
        img_size=rng.uniform(3, 15, n_i),                                             # GB
        img_gpu=rng.permutation(np.r_[np.zeros(n_i // 2), np.ones(n_i - n_i // 2)]).astype(bool),
    )
    world["img_work"] = np.where(world["img_gpu"], rng.uniform(2, 5, n_i), rng.uniform(0.5, 3.0, n_i))
    world["img_mem"] = np.where(world["img_gpu"], rng.uniform(2, 8, n_i), rng.uniform(1, 10, n_i))
    return {k: v.astype(np.float32) if v.dtype == np.float64 else v for k, v in world.items()}


def make_trace(world, split, index, cfg=CFG):
    """One request trace of exactly cfg['tasks'] tasks. Splits use disjoint seeds."""
    rng = np.random.default_rng([cfg["seed"], SPLIT_SEEDS[split], index])
    n_i, T = cfg["images"], cfg["tasks"]
    perm = rng.permutation(n_i)                       # rank -> image
    p_rank = 1 / np.arange(1, n_i + 1) ** cfg["zipf"]
    p_rank /= p_rank.sum()
    rows, slot, recent = [], 0, [int(perm[0])]
    phase = rng.uniform(0, 2 * np.pi)
    while len(rows) < T:
        if slot and slot % cfg["phase_slots"] == 0:   # hot-set drift: swap a few hot and cold ranks
            for _ in range(4):
                a, b = rng.integers(0, 10), rng.integers(10, n_i)
                perm[[a, b]] = perm[[b, a]]
        lam = cfg["rate"] * (1 + 0.4 * np.sin(2 * np.pi * slot / 60 + phase))
        if rng.random() < 0.08:
            lam *= 2.2                                # bursts
        for _ in range(int(np.clip(rng.poisson(lam), 2, 20))):
            if rng.random() < cfg["repeat"]:
                img = recent[rng.integers(len(recent))]
            else:
                img = int(perm[rng.choice(n_i, p=p_rank)])
            recent = (recent + [img])[-5:]
            rows.append((slot, img, world["img_work"][img] * rng.uniform(0.8, 1.2),
                         world["img_mem"][img] * rng.uniform(0.8, 1.2), rng.uniform(0.1, 1.0),
                         rng.uniform(0, 100), rng.uniform(0, 100), rng.uniform(*cfg["allow"])))
        slot += 1
    a = np.array(rows[:T], dtype=np.float64)
    return dict(slot=a[:, 0].astype(np.int32), img=a[:, 1].astype(np.int32), work=a[:, 2], mem=a[:, 3],
                data=a[:, 4], x=a[:, 5], y=a[:, 6], allow=a[:, 7])


def make_dataset(root, counts=(256, 8, 16), cfg=CFG):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    world = make_world(cfg)
    np.savez(root / "world.npz", **world)
    for split, count in zip(("train", "val", "test"), counts):
        traces = [make_trace(world, split, i, cfg) for i in range(count)]
        stacked = {k: np.stack([t[k] for t in traces]).astype(np.float32 if traces[0][k].dtype == np.float64 else np.int32)
                   for k in traces[0]}
        np.savez(root / f"{split}.npz", **stacked)
    (root / "meta.json").write_text(json.dumps(dict(cfg=cfg, counts=dict(zip(("train", "val", "test"), counts)),
                                                    disks=DISKS, policies=POLICIES), indent=1))
    return world


def load(root, split):
    root = Path(root)
    with np.load(root / f"{split}.npz") as f:
        traces = {k: f[k] for k in f.files}
    with np.load(root / "world.npz") as f:
        world = {k: f[k] for k in f.files}
    meta = json.loads((root / "meta.json").read_text())
    return world, traces, meta
