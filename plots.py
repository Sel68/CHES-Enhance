"""Convergence, losses and test-result figures from the CSVs written by run.py (PNG + PDF)."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POLICIES = ("LRU", "LFU", "FIFO", "GDSF")
DISKS = ("tight", "medium", "large")
# Fixed categorical order (colorblind-validated reference palette); colour follows the entity everywhere.
METHOD_COLOR = {"Agent": "#2a78d6", "Greedy_latency": "#eb6834", "Image_priority": "#1baf7a",
                "Least_load": "#eda100", "Greedy_resources": "#e87ba4", "Random": "#6250d6"}
POLICY_COLOR = {"LRU": "#2a78d6", "LFU": "#eb6834", "FIFO": "#1baf7a", "GDSF": "#6250d6"}
DISK_STYLE = {"tight": "-", "medium": "--", "large": ":"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABEL = {"avg_latency": "avg latency (slots)", "completion": "on-time ratio", "hit_rate": "cache hit rate",
         "evictions": "evictions / episode", "reward": "reward (deadline slack / 5)", "avg_wait": "avg image wait (slots)"}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "lines.linewidth": 2, "figure.dpi": 130, "savefig.bbox": "tight"})


def read(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def save(fig, root, name):
    d = Path(root) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(d / f"{name}.{ext}")
    plt.close(fig)


def series(rows, policy, disk, key, x="episode"):
    """-> episodes, array[seed, point] for one config."""
    by_seed = defaultdict(dict)
    for r in rows:
        if r["policy"] == policy and r["disk"] == disk:
            by_seed[r["seed"]][r[x]] = r[key]
    if not by_seed:
        return None, None
    xs = sorted(set.intersection(*[set(d) for d in by_seed.values()]))
    return np.array(xs), np.array([[d[e] for e in xs] for d in by_seed.values()])


def smooth(y, k):
    if k <= 1 or y.shape[-1] < k:
        return y
    kern = np.ones(k) / k
    pad = np.concatenate([np.repeat(y[..., :1], k - 1, -1), y], -1)
    return np.apply_along_axis(lambda r: np.convolve(r, kern, "valid"), -1, pad)


def greedy_ref(base, p, d, key):
    g = [r[key] for r in base if r["policy"] == p and r["disk"] == d and r["method"] == "Greedy_latency"]
    return float(np.mean(g)) if g else np.nan


def convergence(root, val, base, key="avg_latency"):
    """3x4 grid (disk x policy): validation curve, mean +- sd over seeds, individual seeds, greedy reference."""
    fig, axes = plt.subplots(3, 4, figsize=(13, 8.2), sharex=True)
    for i, d in enumerate(DISKS):
        for j, p in enumerate(POLICIES):
            ax = axes[i, j]
            xs, ys = series(val, p, d, key)
            if xs is None:
                continue
            g = greedy_ref(base, p, d, key)
            for y in ys:
                ax.plot(xs, y, color=METHOD_COLOR["Agent"], lw=.7, alpha=.35)
            m, s = ys.mean(0), ys.std(0)
            ax.fill_between(xs, m - s, m + s, color=METHOD_COLOR["Agent"], alpha=.15, lw=0)
            ax.plot(xs, m, color=METHOD_COLOR["Agent"], label=f"CHES agent (mean ± sd, {len(ys)} seeds; thin = seeds)")
            ax.axhline(g, color=METHOD_COLOR["Greedy_latency"], ls="--", lw=1.4, label="Greedy_latency (best heuristic)")
            rest = np.r_[m[1:], g]
            lo, hi = rest.min(), rest.max()
            pad = (hi - lo) * .6 + abs(g) * .02
            ax.set_ylim(lo - pad, hi + pad)
            ax.text(.98, .95, f"untrained: {m[0]:.2f}", transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=7.5)
            ax.set_title(f"{p} · {d} disk", color=INK, fontsize=10)
            if i == 2:
                ax.set_xlabel("training episodes")
            if j == 0:
                ax.set_ylabel(LABEL[key])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(.5, -.02))
    fig.suptitle(f"Convergence on held-out validation traces: {LABEL[key]}", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, .03, 1, .97))
    save(fig, root, f"convergence_{key}")


def convergence_ratio(root, val, base):
    """Headline figure: agent latency / Greedy_latency latency over training, one line per policy, per disk."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    for ax, d in zip(axes, DISKS):
        for p in POLICIES:
            xs, ys = series(val, p, d, "avg_latency")
            if xs is None:
                continue
            r = ys / greedy_ref(base, p, d, "avg_latency")
            m, s = r.mean(0), r.std(0)
            ax.fill_between(xs[1:], (m - s)[1:], (m + s)[1:], color=POLICY_COLOR[p], alpha=.12, lw=0)
            ax.plot(xs[1:], m[1:], color=POLICY_COLOR[p], label=p, marker="o", ms=3)
        ax.axhline(1, color=METHOD_COLOR["Greedy_latency"], ls="--", lw=1.4, label="Greedy_latency = 1")
        ax.set_title(f"{d} disk", color=INK)
        ax.set_xlabel("training episodes")
    axes[0].set_ylabel("agent latency / greedy latency\n(< 1 = agent better)")
    axes[0].set_ylim(.85, 1.12)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, bbox_to_anchor=(.5, -.08))
    fig.suptitle("Validation latency relative to the best heuristic (mean ± sd over 3 seeds; untrained point ≈ 5-9× omitted)",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, .04, 1, .94))
    save(fig, root, "convergence_vs_greedy")


def training_overview(root, train):
    """Paper Fig. 4 / Fig. 9 style: reward and losses for all 12 agents on one set of axes."""
    keys = ("reward", "critic_loss", "actor_loss", "entropy", "alpha", "hit_rate")
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, key in zip(axes.flat, keys):
        for p in POLICIES:
            for d in DISKS:
                xs, ys = series(train, p, d, key)
                if xs is None:
                    continue
                ax.plot(xs, smooth(ys, 5).mean(0), color=POLICY_COLOR[p], ls=DISK_STYLE[d], lw=1.4,
                        label=f"{p} / {d}" if key == "reward" else None)
        ax.set_title({"reward": "training reward per task", "critic_loss": "critic loss (Huber)", "actor_loss": "policy loss",
                      "entropy": "policy entropy", "alpha": "temperature α", "hit_rate": "training cache hit rate"}[key], color=INK)
        ax.set_xlabel("training episodes")
        if key == "alpha":
            ax.set_yscale("log")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=6, bbox_to_anchor=(.5, -.06))
    fig.suptitle("Training dynamics (colour = eviction policy, line style = disk size; 5-iteration moving average, seed mean)",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, .03, 1, .96))
    save(fig, root, "training_dynamics")


def results(root, res):
    methods = [m for m in METHOD_COLOR if any(r["method"] == m for r in res)]
    keys = ("avg_latency", "completion", "hit_rate", "evictions")
    fig, axes = plt.subplots(3, 4, figsize=(14, 8.5))
    width = .8 / len(methods)
    for i, d in enumerate(DISKS):
        for j, key in enumerate(keys):
            ax = axes[i, j]
            for k, m in enumerate(methods):
                vals, errs = [], []
                for p in POLICIES:
                    v = [r[key] for r in res if r["method"] == m and r["policy"] == p and r["disk"] == d]
                    vals.append(np.mean(v) if v else np.nan)
                    errs.append(np.std(v) / np.sqrt(len(v)) if v else 0)
                ax.bar(np.arange(4) + (k - (len(methods) - 1) / 2) * width, vals, width * .9, yerr=errs,
                       color=METHOD_COLOR[m], label=m, error_kw=dict(lw=.8, ecolor=MUTED), edgecolor="white", linewidth=.6)
            ax.set_xticks(range(4), POLICIES)
            if key == "avg_latency":
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
                ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            if key == "evictions" and not np.nansum([r[key] for r in res if r["disk"] == d]):
                ax.text(.5, .5, "0 evictions\n(disk holds the whole catalogue)", transform=ax.transAxes, ha="center",
                        va="center", color=MUTED)
                ax.set_yticks([])
            if i == 0:
                ax.set_title(LABEL[key], color=INK)
            if j == 0:
                ax.set_ylabel(f"{d} disk", color=INK, fontsize=10)
            ax.grid(axis="x", visible=False)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=6, bbox_to_anchor=(.5, -.03))
    fig.suptitle("Test traces: CHES agent vs heuristics (mean ± s.e. over traces and seeds)", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, .03, 1, .97))
    save(fig, root, "results")


def improvement(root, res):
    """Agent vs Greedy_latency, paired by trace: % latency reduction and on-time gain."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    out = {}
    for ax, (key, title, scale) in zip(axes, (("avg_latency", "latency reduction vs Greedy_latency (%)", 1),
                                              ("completion", "on-time ratio gain vs Greedy_latency (pp)", 100))):
        M = np.full((3, 4), np.nan)
        for i, d in enumerate(DISKS):
            for j, p in enumerate(POLICIES):
                g = {r["trace"]: r[key] for r in res if r["method"] == "Greedy_latency" and r["policy"] == p and r["disk"] == d}
                a = defaultdict(list)
                for r in res:
                    if r["method"] == "Agent" and r["policy"] == p and r["disk"] == d:
                        a[r["trace"]].append(r[key])
                if g and a:
                    am = np.array([np.mean(a[t]) for t in g])
                    gm = np.array(list(g.values()))
                    M[i, j] = 100 * (gm - am).mean() / gm.mean() if key == "avg_latency" else scale * (am - gm).mean()
        lim = np.nanmax(np.abs(M)) or 1
        im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim)
        for i in range(3):
            for j in range(4):
                ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=10,
                        color="white" if abs(M[i, j]) > .55 * lim else INK)
        ax.set_xticks(range(4), POLICIES)
        ax.set_yticks(range(3), DISKS)
        ax.grid(False)
        ax.set_title(title, color=INK)
        fig.colorbar(im, ax=ax, shrink=.8)
        out[key] = M.tolist()
    fig.suptitle("Positive (blue) = agent better than the best heuristic", color=MUTED, fontsize=9)
    fig.tight_layout()
    save(fig, root, "improvement_vs_greedy")
    return out


def eviction_pressure(root, res):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, key in zip(axes, ("evictions", "hit_rate", "avg_latency")):
        for p in POLICIES:
            for m, ls in (("Agent", "-"), ("Greedy_latency", "--")):
                ys = [np.mean([r[key] for r in res if r["method"] == m and r["policy"] == p and r["disk"] == d] or [np.nan]) for d in DISKS]
                ax.plot(DISKS, ys, color=POLICY_COLOR[p], ls=ls, marker="o", ms=5, lw=1.6, label=f"{p} · {m}")
        ax.set_title(LABEL[key], color=INK)
        ax.set_xlabel("disk size")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(.5, -.12))
    fig.suptitle("Eviction pressure vs disk size (solid = agent, dashed = Greedy_latency)", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, .05, 1, .95))
    save(fig, root, "eviction_pressure")


def plot_all(root):
    root = Path(root)
    val, train = read(root / "validation.csv"), read(root / "training.csv")
    base, res = read(root / "validation_baselines.csv"), read(root / "results.csv")
    if val:
        convergence_ratio(root, val, base)
        convergence(root, val, base, "avg_latency")
        convergence(root, val, base, "completion")
        convergence(root, val, base, "reward")
    if train:
        training_overview(root, train)
    if res:
        results(root, res)
        eviction_pressure(root, res)
        imp = improvement(root, res)
        (root / "figures" / "improvement_vs_greedy.json").write_text(json.dumps(dict(rows=DISKS, cols=POLICIES, **imp), indent=1))
    print(f"Figures written to {root / 'figures'}", flush=True)


if __name__ == "__main__":
    import sys
    plot_all(sys.argv[1] if len(sys.argv) > 1 else "outputs/run")
