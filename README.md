# CHES with finite disks and eviction policies (JAX)

Rebuild of **CHES** — *Microservice Scheduling With Spatiotemporal Learning in Dynamic and GPU-CPU
Heterogeneous Edge Environment* (Chen et al., IEEE TSC 2026) — extended with **limited per-node disk
space** and four **image eviction policies (LRU, LFU, FIFO, GDSF)**. One CHES agent (CNN + GRU + masked
discrete SAC) is trained per *(eviction policy × disk size)* = **12 configurations**, all in one compiled
JAX program on CPU or GPU. The authors' original code is kept untouched in `code available/`.

```
data.py      world (20 nodes, 50 images/services) + train/val/test request traces (.npz)
env.py       JAX simulator: latency model, serial image pulls, disks, LRU/LFU/FIFO/GDSF, heuristics, replay export
agent.py     CNN-GRU actor + twin critics, masked discrete SAC, on-device replay buffer
run.py       CLI: data | train | evaluate | plot | smoke   (progress + ETA printed every iteration)
plots.py     convergence, training dynamics, test results, eviction pressure, improvement heat-map
viz.py       animation from replay JSON only: MP4 (GIF fallback) + self-contained HTML player
test_core.py simulator invariants (capacity, pinning, eviction order vs. reference, type constraints)
```

## Quick start

```bash
pip install -r requirements.txt                 # CPU
pip install -U "jax[cuda12]"                    # NVIDIA GPU (use cuda13 for newest drivers / WSL)

python run.py smoke --device cpu                # ~1 min end-to-end check: tiny data, 12 agents, plots, 3 animations
python run.py data  --data data                 # 256 train / 8 val / 16 test traces, 2000 tasks each
python run.py train --data data --out outputs/run --device gpu --episodes 800 --seeds 0 1 2
python run.py evaluate --data data --out outputs/run --device gpu   # test metrics + replay logs
python run.py plot --out outputs/run
python viz.py --scenarios --root outputs/run --mode both --fps 10  # the 3 showcase animations (200 slots)
python viz.py --replay outputs/run/replays/Agent_LRU_tight_test0.json --mode mp4|html --fps 10
python -m pytest -q test_core.py
```

`--device auto|cpu|gpu` works everywhere; the same code and checkpoints run on CPU and GPU. On Windows,
JAX GPU runs inside WSL2 (this project was trained on an RTX 5060 Laptop GPU via WSL, `jax[cuda13]`).
Training prints one line per iteration (episodes done, s/it, elapsed, **ETA**, train latency / on-time /
hit rate, critic loss, temperature, entropy) and a validation table (agent vs Greedy_latency) every
`--eval_every` iterations; the same lines go to `<out>/progress.log`.

## What was rebuilt, and why

The previous attempt in this folder (Python event simulator + JAX agent) needed ~38 s per training
episode on the GPU and, after 10 episodes, its agent was still worse than Greedy_latency. The author code
(`code available/`) has no eviction, discards tasks when no node fits, and its configs do not match the
paper. So the environment was rewritten as **pure JAX functions** (`jit`/`vmap`/`scan`): rollouts of all
agents × parallel environments and SAC updates run in a single compiled program, ~0.1 s per agent-episode
on the laptop GPU (≈ 300× faster), which is what makes 12 configs × 3 seeds × 800 episodes feasible.

### Environment (paper Sec. III, authors' code units: time in slots)

| Item | Choice |
|---|---|
| Nodes | 20 = 14 CPU + 6 GPU (paper default 20), random positions in 100×100, ordered CPU→GPU then by x so the 1-D CNN sees spatial neighbours |
| Node resources | CPU 3–6.5 GHz, RAM 80–180 GB; GPU 10–16 (normalised FLOPS), video memory 16–32 GB, GPU mem bandwidth 2–6; network 2–6 Gbps (paper Sec. V-A) |
| Images / services | 50 images, 3–15 GB (paper); each image is one microservice, ½ CPU-, ½ GPU-preferred |
| Latency (Eq. 1–6) | `T = max(T_upload, T_image) + T_comp`; `T_image` = 0 if cached, residual pull if downloading, else node pull-queue + `size/bw`; CPU: `c·U/F`; GPU: `α·d·U/Bg + f·U/(Fg(1-λ)^(U-1))`, λ = 0.05, α = 0.5; a GPU service on a CPU node needs 4.5× the work (authors' code); CPU services cannot run on GPU nodes |
| Concurrency | U = jobs on the node at admission (incl. the new one), as in Eq. 4–5; jobs hold memory and pin their image from admission to finish; ≤ 16 jobs/node |
| Pulls | serial per node (paper's download queue); duplicate requests share an in-flight pull ("coalesced") |
| Feasibility mask (Alg. 2) | type, memory, job slot, and disk: image cached **or** `pinned/downloading bytes + size ≤ capacity`; if no node is feasible the task is rejected (latency = deadline + 5 slots) |
| Reward (Eq. 18) | `(deadline allowance − latency) / 5`, immediate (the model's latency is known at admission) |

**Disk regimes** (per-node capacity as a fraction of the 459 GB catalogue, calibrated with Greedy_latency):
**tight 5 %** (≈ 2 images per node, ~500 evictions per 2000-task trace), **medium 10 %** (≈ 4–5 images,
~70 evictions), **large 100 %** (everything fits, never evicts → the original paper's setting).

**Eviction policies** (only unpinned, fully downloaded images are candidates; exact greedy eviction in
policy order until the new image fits): **LRU** least-recent access; **LFU** fewest references while
resident (ties → LRU); **FIFO** oldest insertion; **GDSF** smallest `H = L + freq·cost/size` with
`cost = 1 + pull time` and per-node inflation `L ← max evicted H` (ties → LRU).

### Data (train/val/test)

`data.py` generates one fixed world and disjoint request traces (separate seed streams per split):
256 train, 8 validation (model selection), 16 test traces, 2000 tasks each (~145 slots). Arrivals are
Poisson with a sinusoidal rate (mean 14 tasks/slot) plus 8 % bursts (×2.2). **Services recur**: image
popularity is Zipf(1.0) over a shuffled ranking whose hot set drifts every 40 slots (4 hot/cold swaps),
plus 20 % short-term repeats of one of the last 5 requested images — so an evicted image is often needed
again soon, which is exactly where disk size and the eviction policy matter. Deadlines are 2.5–5 slots
after arrival (authors: 3–5). Work and memory have ±20 % jitter per request.

### Agent (paper Sec. IV)

Observation per node (29 features): node static (type, speed, bandwidth, capacity, position), node
dynamic (load, memory, disk use, pull backlog, popularity-weighted cache value), task × node (image
ready / downloading, estimated wait, compute, upload and total latency, gap to the best node, bytes
that would be evicted, popularity of the would-be victims, #victims, feasibility), task (type, work,
memory, deadline, image size, popularity, replication). Network: shared per-node encoder (1×1 conv) →
**kernel-3 1-D CNN across nodes** (spatial) for the current task; **GRU over the last 4 scheduled tasks**
(per-step pooled cluster embedding; temporal coupling within/after a slot, Fig. 2); per-node head
`[CNN, encoder, GRU] → 64 → 16 → score` plus a forced-rejection logit. Actor, two critics and two target
critics have the same architecture. **Masked discrete SAC** (Eq. 23–29): exact expectation over valid
actions, twin-min targets, Huber critic loss, auto-tuned temperature (target entropy 0.1·log #valid
nodes), τ = 0.005, actor/critic lr 1e-4 / 3e-4 (paper), batch 256, replay 32 000 transitions per agent
(paper 30 000), 8 parallel envs, 8 updates every 50 env steps. No behaviour cloning or heuristic is used
inside the agent — it learns from scratch. Evaluation is greedy (argmax of the actor). The best
checkpoint per agent is selected on the validation traces; test traces are never seen in training.

**Known-reward critic.** The immediate reward of *every* action (Eq. 18) is computable from the state
(the latency estimate equals the realised latency; `test_core.py` checks this), so the soft Q-function is
written exactly as `Q(s,a) = R(s,a) + F(s,a)` and the twin critics learn only the future part `F` with
target `γ·V(s')`, `V(s') = Σ π(a'|s')[R(s',a') + min F_targ(s',a') − α log π(a'|s')]`. Same Bellman
equation, less regression noise. `--plain_critic` restores the standard critic.

**Convergence fixes (and why γ = 0.9, not the paper's 0.98).** The first full run (paper settings,
target entropy 0.3·log #valid, kept in `outputs/run_v1_ent0.3`) beat Greedy_latency on tight disks but
large-disk agents drifted from 1.01× to 1.11× greedy latency while α and the critic loss kept rising.
Single-seed 240-episode checks (validation latency / Greedy_latency, mean of 4 policies):

| variant | tight | medium | large (trend over training) | critic loss |
|---|---|---|---|---|
| paper γ 0.98, entropy 0.3 | 0.94 | 0.97 | 1.05 → 1.11 drifting | rising |
| γ 0.98, entropy 0.1 | 0.92 | 0.99 | 1.01 → 1.09 drifting | 0.16 → 0.45 |
| + known-reward critic | 0.94 | 0.96 | 1.04 → 1.11 drifting | 0.16 → 0.47 |
| + γ 0.95 | 0.92 | 0.95 | 1.02 → 1.07 slow drift | 0.14 → 0.25 |
| **+ γ 0.9 (used)** | **0.94** | **0.98** | **≈ 1.02 flat** | **≈ 0.13 flat** |

With γ = 0.98 the future term is ~50 tasks long and its bootstrapped noise exceeds the small immediate
differences between near-equivalent nodes, which matters most on a large disk; the actor then follows
noise (actor-argmax, actor-sampling and critic-argmax all degraded equally, so it is not an argmax
artefact). γ = 0.9 (≈ 10-task horizon, about one slot) keeps the eviction benefits and converges cleanly.

**12 configurations × 3 seeds = 36 independently trained networks**, vmapped in one program; seeds give
the ± bands in the convergence plots.

### Baselines (paper Sec. V-A.2)

Greedy_latency (lowest estimated latency incl. pull wait — the strongest heuristic), Image_priority
(cached image first, else random), Least_load (fewest jobs), Greedy_resources (highest free speed ×
memory share), Random. All share the same physics, masks and traces.

## Outputs

`outputs/<run>/`: `training.csv` (per iteration × agent: rollout metrics + losses), `validation.csv`,
`validation_baselines.csv`, `results.csv` (test, per trace), `summary.json`, `models.pkl` (best and last
parameters), `progress.log`, `replays/*.json`, and `figures/`:

| Figure | Content |
|---|---|
| `convergence_vs_greedy` | headline: validation latency ÷ Greedy_latency over training, one line per policy, per disk, ± sd over seeds |
| `convergence_avg_latency` / `_completion` / `_reward` | 3×4 grid (disk × policy): validation curve, mean ± sd over seeds, individual seeds, Greedy_latency reference, untrained value |
| `training_dynamics` | reward, critic loss, policy loss, entropy, α, hit rate for all 12 agents (paper Fig. 4 / 9 style) |
| `results` | test latency (log), on-time ratio, hit rate, evictions for all methods × policies × disks |
| `eviction_pressure` | evictions / hit rate / latency vs disk size per policy, agent vs Greedy_latency |
| `improvement_vs_greedy` | heat-map of % latency reduction and on-time gain vs Greedy_latency (paired by trace) |
| `replay_*.mp4` / `.html` | the three animation scenarios |

### Animation / replay

During `evaluate`, every method × config is replayed on test trace 0 and logged to
`replays/<method>_<policy>_<disk>_test0.json` (first `--replay_slots` slots, default 200): static nodes
(x, y, type), capacity and image sizes, and per task id, slot, type, image, chosen node, hit / coalesced
/ miss / rejected, evicted images, pull start/finish, finish time, deadline and outcome. `viz.py` reads
only these files (no simulator, JAX or GPU). Scenes: CPU node = square, GPU node = hexagon, disk bar of
coloured image blocks (width = size, white→colour fill = pull progress, black outline = pinned); tasks
(circle = CPU service, triangle = GPU service) fly from their origin to the chosen node; green ring =
hit, orange arc = pull progress, evicted blocks fade red with the victim id, late tasks turn red. Each
panel shows cumulative avg latency, hit rate, evictions, on-time ratio and a rolling-hit-rate sparkline;
colours per image are identical across panels. Node positions are spread slightly for display only.
Scenarios: (1) `agent_vs_greedy` — agent vs Greedy_latency, LRU, tight disk; (2) `four_policies` — the
agent under LRU / LFU / FIFO / GDSF at tight disk (each policy's own trained network); (3)
`disk_pressure` — large vs tight disk, LRU agent. The HTML is a single file (inline data + canvas JS,
play/pause, seek slider, speed, light/dark). Test trace 0 spans 140 slots, so each video is ~56 s.

Note on the showcase: replays use seed 0 on test trace 0. That trace is a hard one for the LRU agent
(seed 0 beats greedy on 8/16 test traces under LRU and loses on trace 0: 2.01 vs 1.89), so
`replay_agent_vs_greedy` shows the agent slightly behind. `replay_agent_vs_greedy_FIFO` is a more
typical case (seed 0 FIFO agent beats greedy on 13/16 test traces, trace 0: 1.87 vs 1.99). Averages
over all traces and seeds are in the Results section. Any pair can be rendered with
`python viz.py --replay outputs/run/replays/Agent_<P>_<D>_test0.json outputs/run/replays/Greedy_latency_<P>_<D>_test0.json --mode both`.

## Results

Full run: 12 configurations × 3 seeds = 36 agents, 800 episodes each, γ = 0.9, RTX 5060 Laptop GPU,
**59.5 min** of training (`outputs/run`). Test = 16 unseen traces (32 000 tasks); agent numbers are the
mean over 3 seeds of the best-on-validation checkpoint. Latency in slots.

**Convergence.** All 36 agents learn from a random policy (latency 8–17) to below the best heuristic
within ~80 episodes; afterwards validation latency, critic loss (~0.13–0.16), entropy (at its 0.27
target) and α (~0.018) stay flat for 700 episodes with tight seed bands
(`figures/convergence_vs_greedy.png`, `convergence_avg_latency.png`, `training_dynamics.png`).

**Agent vs the best heuristic (Greedy_latency), test traces**

| disk | policy | agent latency | Greedy_latency | Δ latency | agent on-time | greedy on-time | agent hit | greedy hit | agent evictions | greedy evictions |
|---|---|---|---|---|---|---|---|---|---|---|
| tight | LRU | 2.222 | 2.292 | −3.1% | 0.859 | 0.852 | 0.608 | 0.586 | 442 | 498 |
| tight | LFU | 2.215 | 2.266 | −2.2% | 0.860 | 0.854 | 0.612 | 0.591 | 440 | 489 |
| tight | FIFO | 2.238 | 2.361 | −5.2% | 0.858 | 0.844 | 0.600 | 0.566 | 457 | 516 |
| tight | GDSF | 2.179 | 2.295 | −5.0% | 0.863 | 0.847 | 0.619 | 0.587 | 426 | 492 |
| medium | LRU | 1.359 | 1.435 | −5.3% | 0.951 | 0.940 | 0.854 | 0.843 | 127 | 154 |
| medium | LFU | 1.369 | 1.440 | −5.0% | 0.950 | 0.939 | 0.846 | 0.834 | 140 | 167 |
| medium | FIFO | 1.406 | 1.521 | −7.5% | 0.947 | 0.933 | 0.844 | 0.822 | 134 | 175 |
| medium | GDSF | 1.389 | 1.433 | −3.0% | 0.948 | 0.939 | 0.842 | 0.838 | 135 | 160 |
| large | LRU | 1.161 | 1.167 | −0.5% | 0.970 | 0.966 | 0.921 | 0.920 | 0 | 0 |
| large | LFU | 1.164 | 1.167 | −0.3% | 0.969 | 0.966 | 0.920 | 0.920 | 0 | 0 |
| large | FIFO | 1.165 | 1.167 | −0.1% | 0.968 | 0.966 | 0.919 | 0.920 | 0 | 0 |
| large | GDSF | 1.166 | 1.167 | −0.1% | 0.969 | 0.966 | 0.921 | 0.920 | 0 | 0 |

**All methods (mean over the 4 eviction policies)**

| method | tight latency | tight on-time | medium latency | medium on-time | large latency | large on-time |
|---|---|---|---|---|---|---|
| **CHES agent** | **2.21** | **0.860** | **1.38** | **0.949** | **1.16** | **0.969** |
| Greedy_latency | 2.30 | 0.849 | 1.46 | 0.938 | 1.17 | 0.966 |
| Greedy_resources | 4.44 | 0.583 | 6.81 | 0.421 | 2.36 | 0.842 |
| Least_load | 5.11 | 0.526 | 6.74 | 0.411 | 2.98 | 0.778 |
| Image_priority | 8.88 | 0.545 | 8.74 | 0.588 | 8.38 | 0.600 |
| Random | 9.78 | 0.385 | 13.10 | 0.250 | 15.87 | 0.330 |

Findings:
- The agent beats every heuristic in **all 12 configurations**. Its advantage over Greedy_latency
  appears exactly where the disk is constrained (2–8 % lower latency, +0.6–1.6 pp on-time) and vanishes
  when it is not (large disk: −0.1 to −0.5 %, essentially a tie, since greedy is near-optimal without
  eviction).
- The agent **evicts 10–23 % less** and gets higher hit rates than greedy under the same eviction
  policy: it learns where to place images so that popular ones are not pushed out (the would-be-victim
  popularity feature + the critic's future term).
- Eviction policy matters less for the learned scheduler than for greedy: greedy's spread across
  policies on the tight disk is 2.27–2.36, the agent's 2.18–2.24. GDSF is the best policy for the agent
  under heavy eviction pressure; FIFO is the worst for greedy, and the agent recovers most of FIFO's loss.
- Heuristics that ignore image state (Least_load, Greedy_resources, Random) collapse as disks shrink,
  because they trigger many cold pulls (≈ 850–900 evictions per trace on the tight disk).
- As in the paper, Greedy_latency is the strongest heuristic and Image_priority (cached-first) is poor
  because it ignores heterogeneous compute and congestion.

The first full run with the paper's γ = 0.98 is kept in `outputs/run_v1_ent0.3` (stopped at 288
episodes) and the 240-episode single-seed checks in `outputs/check_*`, for the comparison above.

## Limitations

Synthetic workloads and normalised compute units (as in the paper); compute time is fixed at admission
from the concurrency at that moment (paper Eq. 4–5) rather than re-shared when later jobs arrive; no
layer sharing, registry congestion or cloud offload; generalisation is to new request sequences on a
fixed topology and catalogue.
