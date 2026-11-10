"""Publication-style figures and honest seed/trace bootstrap summaries (no JAX)."""
import csv
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ches-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from data import DISKS, POLICIES, write_json

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#6E7884"]


def read_rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def interval(rows, metric, rng, draws=1000):
    """Crossed bootstrap: independently resample training seeds and shared traces."""
    seeds, traces = sorted({r["seed"] for r in rows}), sorted({r["trace"] for r in rows})
    values = {(r["seed"], r["trace"]): float(r[metric]) for r in rows}
    matrix = np.array([[values[s, t] for t in traces] for s in seeds])
    sample_s = rng.integers(len(seeds), size=(draws, len(seeds)))
    sample_t = rng.integers(len(traces), size=(draws, len(traces)))
    estimates = matrix[sample_s[:, :, None], sample_t[:, None, :]].mean((1, 2))
    return float(matrix.mean()), *np.quantile(estimates, [.025, .975]).tolist()


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.labelcolor": "#374151", "text.color": "#1B2535", "axes.edgecolor": "#D1D5DB",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#E5E7EB", "grid.alpha": .7,
                         "figure.facecolor": "#FAFBFD", "axes.facecolor": "white", "savefig.dpi": 180})


def save(fig, root, name):
    fig.savefig(root / f"{name}.png", bbox_inches="tight")
    fig.savefig(root / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_all(output):
    output = Path(output)
    root = output / "figures"
    root.mkdir(parents=True, exist_ok=True)
    style()
    validation = read_rows(output / "validation.csv")
    training = read_rows(output / "training.csv") if (output / "training.csv").exists() else []
    results = read_rows(output / "results.csv")
    smoke = any(r["smoke"] == "True" for r in results)
    tag = "CPU SMOKE · mechanics only" if smoke else "Held-out evaluation"
    rng = np.random.default_rng(2026)
    policies = [p for p in POLICIES if any(r["policy"] == p for r in validation)]
    disks = [d for d in DISKS if any(r["disk"] == d for r in validation)]
    fig, axes = plt.subplots(len(policies), len(disks), figsize=(4.4 * len(disks), 2.8 * len(policies)), squeeze=False, constrained_layout=True)
    convergence = []
    for pi, policy in enumerate(policies):
        for di, disk in enumerate(disks):
            ax = axes[pi, di]
            subset = [r for r in validation if r["policy"] == policy and r["disk"] == disk]
            episodes = sorted({int(r["episode"]) for r in subset})
            means = []
            for seed in sorted({r["seed"] for r in subset}):
                rows = sorted([r for r in subset if r["seed"] == seed], key=lambda r: int(r["episode"]))
                ax.plot([int(r["episode"]) for r in rows], [float(r["reward"]) for r in rows], color=COLORS[0], alpha=.18, lw=1)
            for ep in episodes:
                means.append(np.mean([float(r["reward"]) for r in subset if int(r["episode"]) == ep]))
            stds = [np.std([float(r["reward"]) for r in subset if int(r["episode"]) == ep]) for ep in episodes]
            ax.plot(episodes, means, color=COLORS[0], lw=2, label="Validation mean")
            ax.fill_between(episodes, np.array(means) - stds, np.array(means) + stds, color=COLORS[0], alpha=.12, label="±1 seed SD")
            # Baseline is evaluated on validation, never borrowed from test results.
            baseline_path = output / "validation_baselines.csv"
            if baseline_path.exists():
                baselines = [float(r["reward"]) for r in read_rows(baseline_path) if r["policy"] == policy and r["disk"] == disk]
                if baselines:
                    ax.axhline(np.mean(baselines), color=COLORS[1], ls="--", label="Greedy validation")
            ax.set_title(f"{policy} · {disk} disk")
            ax.set_xlabel("Training episode"); ax.set_ylabel("Mean deadline slack / 50 s")
            if pi == 0 and di == 0:
                ax.legend(fontsize=8)
            # A diagnostic, not a proof of convergence or a stopping rule.
            tail = max(3, len(episodes) // 4)
            slope = float(np.polyfit(episodes[-tail:], means[-tail:], 1)[0]) if len(episodes) >= 3 else None
            convergence.append(dict(policy=policy, disk=disk, validation_points=len(episodes),
                seeds=len({r["seed"] for r in subset}), final_reward=float(means[-1]),
                initial_reward=float(means[0]), tail_slope_per_episode=slope,
                status="insufficient evidence" if smoke or len(episodes) < 20 or len({r['seed'] for r in subset}) < 3 else "inspect raw curves and tail slope; no automatic convergence claim"))
    fig.suptitle(f"Learning stability | {tag}", fontsize=17)
    save(fig, root, "convergence")
    write_json(output / "convergence_diagnostics.json", convergence)

    methods = [m for m in ("Agent", "Greedy_latency", "Image_priority", "Least_load", "Greedy_resources", "Random") if any(r["method"] == m for r in results)]
    summaries = []
    for policy in policies:
        for disk in disks:
            for method in methods:
                rows = [r for r in results if r["policy"] == policy and r["disk"] == disk and r["method"] == method]
                if not rows:
                    continue
                entry = dict(policy=policy, disk=disk, method=method, seeds=len({r["seed"] for r in rows}), traces=len({r["trace"] for r in rows}))
                for metric in ("avg_latency", "p95_latency", "completion_ratio", "hit_rate", "evictions", "rejection_rate", "avg_download_wait"):
                    mean, lo, hi = interval(rows, metric, rng)
                    entry[metric] = dict(mean=mean, low=lo, high=hi)
                summaries.append(entry)
    write_json(output / "summary.json", dict(smoke=smoke, uncertainty="95% crossed seed/trace percentile bootstrap; single seed or trace cannot estimate that variance component", results=summaries))
    metrics = [("avg_latency", "Mean latency (s) ↓"), ("completion_ratio", "On-time fraction ↑"), ("hit_rate", "Cache hit fraction ↑"), ("evictions", "Evictions per trace ↓")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    labels = [f"{p}\n{d}" for d in disks for p in policies]
    for ax, (metric, label) in zip(axes.flat, metrics):
        x = np.arange(len(labels)); width = .8 / len(methods)
        for mi, method in enumerate(methods):
            entries = [next(r for r in summaries if r["method"] == method and r["policy"] == p and r["disk"] == d)[metric] for d in disks for p in policies]
            means = np.array([e["mean"] for e in entries])
            yerr = np.maximum(0, np.array([means - [e["low"] for e in entries], [e["high"] for e in entries] - means]))
            ax.bar(x + (mi - (len(methods) - 1) / 2) * width, means, width, color=COLORS[mi], label=method.replace("_", " "), yerr=yerr, capsize=1.5, zorder=3)
        ax.set_xticks(x, labels, fontsize=8); ax.set_ylabel(label)
        if metric in ("completion_ratio", "hit_rate"):
            ax.set_ylim(0, 1.05)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(f"Scheduling × eviction × disk budget | {tag}", fontsize=17)
    save(fig, root, "results")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    for ax, metric, label in zip(axes, ("critic_loss", "entropy", "alpha"), ("Twin-critic Huber loss", "Policy entropy", "Entropy temperature")):
        for pi, policy in enumerate(policies):
            rows = [r for r in training if r["policy"] == policy and r["disk"] == disks[0]]
            eps = sorted({int(r["episode"]) for r in rows})
            ax.plot(eps, [np.mean([float(r[metric]) for r in rows if int(r["episode"]) == e]) for e in eps], color=COLORS[pi], label=policy, lw=2)
        ax.set_xlabel("Training episode"); ax.set_ylabel(label)
    axes[0].legend(frameon=False)
    fig.suptitle(f"Optimizer diagnostics · {disks[0]} disk | {tag}", fontsize=16)
    save(fig, root, "diagnostics")
    # Paired differences retain common test-trace difficulty for the main comparison.
    paired = []
    for policy in policies:
        for disk in disks:
            greedy = {r["trace"]: float(r["avg_latency"]) for r in results if r["method"] == "Greedy_latency" and r["policy"] == policy and r["disk"] == disk}
            rows = [dict(r, delta=str(float(r["avg_latency"]) - greedy[r["trace"]])) for r in results if r["method"] == "Agent" and r["policy"] == policy and r["disk"] == disk]
            if rows:
                mean, lo, hi = interval(rows, "delta", rng)
                paired.append(dict(policy=policy, disk=disk, agent_minus_greedy_seconds=mean, low=lo, high=hi))
    write_json(output / "paired_comparisons.json", paired)
    print(f"Figures and bootstrap summaries: {root}", flush=True)


def compare_validation(roots, labels, output):
    """Compare recorded validation curves; never use test results for tuning."""
    style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for index, (root, label) in enumerate(zip(roots, labels)):
        rows = read_rows(Path(root) / "validation.csv")
        episodes = sorted({int(r["episode"]) for r in rows})
        for ax, metric, ylabel in zip(axes, ("avg_latency", "completion_ratio"), ("Validation mean latency (s) ↓", "Validation on-time fraction ↑")):
            values = [[float(r[metric]) for r in rows if int(r["episode"]) == ep] for ep in episodes]
            mean, std = np.array([np.mean(v) for v in values]), np.array([np.std(v) for v in values])
            ax.plot(episodes, mean, color=COLORS[index], lw=2, label=label)
            ax.fill_between(episodes, mean - std, mean + std, color=COLORS[index], alpha=.12)
            ax.set_xlabel("Training episode"); ax.set_ylabel(ylabel)
    baselines = read_rows(Path(roots[-1]) / "validation_baselines.csv")
    for ax, metric in zip(axes, ("avg_latency", "completion_ratio")):
        ax.axhline(np.mean([float(r[metric]) for r in baselines]), color="#374151", ls="--", lw=1.4, label="Greedy_latency")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].set_ylim(0, 1.02)
    fig.suptitle("Training choices · fixed validation traces · mean ± 1 seed SD", fontsize=15)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save(fig, output.parent, output.stem)
    print(f"Validation comparison: {output.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--compare", nargs="+")
    parser.add_argument("--labels", nargs="+")
    args = parser.parse_args()
    if args.compare:
        if not args.labels or len(args.labels) != len(args.compare) or len(args.labels) > len(COLORS):
            parser.error("Supply one --labels value per comparison root, up to six")
        compare_validation(args.compare, args.labels, args.output)
    else:
        plot_all(args.output)
