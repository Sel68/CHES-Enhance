"""Seeded synthetic service catalogue and disjoint, temporally local request traces."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import numpy as np


POLICIES = ("LRU", "LFU", "FIFO", "GDSF")
DISKS = {"tight": 0.045, "medium": 0.10, "large": 1.01}
SPLIT_TAGS = {"train": 11, "val": 22, "test": 333, "calibration": 33}


@dataclass
class Config:
    seed: int = 2026
    cpu_nodes: int = 14
    gpu_nodes: int = 6
    images: int = 50
    slots: int = 400
    slot_seconds: float = 2.0
    min_tasks: int = 2
    max_tasks: int = 12
    phase_slots: int = 60
    gpu_contention: float = 0.05


def world(cfg):
    rng = np.random.default_rng(cfg.seed)
    sizes = rng.uniform(3, 15, cfg.images)
    kinds = rng.integers(0, 2, cfg.images)
    kinds[:2] = [0, 1]
    images = [dict(id=i, size=float(sizes[i]), type="GPU" if kinds[i] else "CPU",
                   work=float(rng.uniform(2, 5) if kinds[i] else rng.uniform(1.5, 2.5)),
                   memory=float(rng.uniform(2, 8))) for i in range(cfg.images)]
    nodes = []
    for i in range(cfg.cpu_nodes + cfg.gpu_nodes):
        gpu = i >= cfg.cpu_nodes
        nodes.append(dict(id=i, type="GPU" if gpu else "CPU",
                          x=float(rng.uniform(7, 93)), y=float(rng.uniform(10, 90)),
                          speed=float(rng.uniform(10, 16) if gpu else rng.uniform(3, 6.5)),
                          memory=float(rng.uniform(16, 32) if gpu else rng.uniform(80, 180)),
                          bandwidth=float(rng.uniform(2, 6))))
    # Stable spatial order makes the CNN neighbourhood well defined.
    nodes.sort(key=lambda n: (n["type"], n["x"]))
    for i, n in enumerate(nodes):
        n["id"] = i
    return dict(config=asdict(cfg), observation_version=2, disk_scales=DISKS.copy(), split_seed_tags=SPLIT_TAGS.copy(), images=images, nodes=nodes)


def trace(catalogue, split, index=0, slots=None):
    """Same services may recur; no request sequences or trace seeds cross splits."""
    if split not in SPLIT_TAGS or index < 0:
        raise ValueError("Use train/val/test/calibration and a nonnegative trace index")
    cfg = Config(**catalogue["config"])
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, catalogue.get("split_seed_tags", SPLIT_TAGS)[split], index]))
    order = rng.permutation(cfg.images)
    base = 1 / np.arange(1, cfg.images + 1) ** 1.1
    tasks, previous = [], int(order[0])
    for slot in range(slots or cfg.slots):
        shift = ((slot // cfg.phase_slots) * 7) % cfg.images
        p = np.empty(cfg.images)
        p[np.roll(order, shift)] = base / base.sum()
        # Slowly varying intensity plus bursts; not IID image selection.
        burst = rng.random() < 0.12
        count = cfg.max_tasks if burst else int(rng.integers(cfg.min_tasks, max(cfg.min_tasks + 1, cfg.max_tasks - 2)))
        for _ in range(count):
            image = previous if rng.random() < 0.25 else int(rng.choice(cfg.images, p=p))
            previous = image
            service = catalogue["images"][image]
            arrival = slot * cfg.slot_seconds
            tasks.append(dict(id=len(tasks), slot=slot, arrival=arrival, image=image,
                              type=service["type"], work=service["work"] * float(rng.uniform(.8, 1.2)),
                              memory=service["memory"], data=float(rng.uniform(.001, .01)),
                              x=float(rng.uniform(0, 100)), y=float(rng.uniform(0, 100)),
                              deadline=arrival + float(rng.uniform(20, 50))))
    return dict(id=f"{split}_{index:04d}", split=split, index=index,
                slots=slots or cfg.slots, tasks=tasks)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False))


def make_dataset(root, cfg, counts=(64, 4, 8)):
    root = Path(root)
    catalogue = world(cfg)
    write_json(root / "world.json", catalogue)
    for split, count in zip(("train", "val", "test"), counts):
        for i in range(count):
            write_json(root / split / f"{i:04d}.json", trace(catalogue, split, i))
    write_json(root / "manifest.json", dict(config=asdict(cfg), counts=dict(zip(("train", "val", "test"), counts)),
                                           generator_version=3, split_seed_tags=SPLIT_TAGS.copy(), observation_version=2))
    return catalogue
