# CHES with finite image caches · JAX

A compact research reconstruction of **Microservice Scheduling With Spatiotemporal Learning in Dynamic and GPU–CPU Heterogeneous Edge Environment** (Chen et al., IEEE TSC, 2026), extended with finite disk capacity and LRU, LFU, FIFO, and GDSF eviction. The supplied PDF and `code available/` are preserved. This implementation does **not** claim to reproduce the paper's published numbers.

Six implementation files, one test file, no PyTorch or Flax:

| File | Purpose |
|---|---|
| `data.py` | Configuration, fixed service catalogue, reproducible request traces |
| `env.py` | Concurrent event simulation, cache policies, observations, evaluation logs |
| `agent.py` | JAX CNN–GRU discrete SAC, replay, trusted local checkpoints |
| `run.py` | Data generation, training, evaluation, controlled replay recording, smoke workflow |
| `plots.py` | Convergence/results/optimizer figures, uncertainty, paired comparisons |
| `viz.py` | JSON-only video/GIF and standalone HTML replay |
| `test_core.py` | Physical/cache invariants and learning/replay checks |

## Run on this CPU machine

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py --smoke --device cpu --output outputs/smoke
```

Use a fresh output directory. Smoke trains all **12 configurations**, evaluates the agent and five heuristics, saves three figure families as PNG/PDF, records replay JSON, renders **all three animation scenarios** as HTML and MP4 (GIF when ffmpeg is unavailable), and runs the tests. It uses 40-slot traces, a small network, two learning episodes, and one seed. Outputs explicitly say **SMOKE; mechanics only**. Re-running a smoke run requires a different `--output`.

On the existing machine `python3` is available; inside a virtual environment `python` works too. Matplotlib uses a temporary writable cache. ffmpeg is optional and is already installed here. HTML export needs only Python's standard library; video export needs matplotlib and Pillow. The HTML works from a local file without a server or network.

## Full experiment on GPU

Generate the dataset once and copy it with the source to the GPU machine:

```bash
python run.py data --data data --slots 400 --train-traces 64 --val-traces 4 --test-traces 8
pip install -r requirements.txt
pip install 'jax[cuda12]==0.6.2'
python run.py train --data data --output outputs/full --device gpu
python run.py evaluate --data data --output outputs/full --device gpu --replay_slots 200
python run.py plot --output outputs/full
python viz.py --scenarios --root outputs/full --fps 10
```

Check NVIDIA driver/hardware requirements against the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html). `--device auto` lets JAX select an available backend; `--device cpu` forces CPU; `--device gpu` (or `--device cuda`) explicitly selects NVIDIA CUDA and fails if CUDA initialization fails. The CLI maps `gpu` to `JAX_PLATFORMS=cuda`: JAX 0.6.2 expands the literal `gpu` platform alias to both CUDA and ROCm and requires both to initialize, which fails on an NVIDIA-only installation. Neural inference and updates use JAX on the selected device; the discrete-event simulator and replay sampling run on CPU. Small experiments can be slower on GPU because of dispatch overhead. No GPU execution has been verified on this CPU-only machine.

### Colab and the VS Code extension

Open `colab.ipynb` locally in VS Code and select a **Colab GPU kernel**. The notebook embeds the nine source/support files; running its first code cell transfers them to the remote runtime. It includes dependency repair, a GPU compilation check, persistent Drive data/results, a single-configuration trial, evaluation and all replay exports. No GitHub remote is required. The [official extension](https://github.com/googlecolab/colab-vscode) connects notebook cells to a remote runtime; local Python files are not automatically synchronized.

For occasional manual transfers, the extension also provides **right-click files/folders → Upload to Colab**. Its experimental **Mount Server To Workspace** command lets VS Code edit remote files directly; this is distinct from synchronizing this local checkout. See the [official user guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide#working-with-files).

After source changes, run this in your **local terminal**:

```bash
python3 colab.py
```

Accept VS Code's notebook reload if prompted, then rerun its source cell and configuration cell. The generator replaces only the source cell, preserving your settings, other cells and saved outputs. Each snapshot gets a different remote directory, so an existing training process retains its original sources. Restart training to use an updated snapshot. Dependency changes also require rerunning installation. Data and checkpoints remain on Drive; source snapshots are archived there for reproducibility. This is explicit source transfer through a notebook cell, not a live filesystem mount.

If Colab has preinstalled CUDA plugins, remove both plugin families **before** installing this project's pinned CUDA 12 stack. Mixed versions can cause `register_custom_type_handler`, `PJRT_Api already exists`, or fatal PJRT structure-size errors:

```python
%pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt jax-cuda13-plugin jax-cuda13-pjrt
%pip install --no-cache-dir -r requirements.txt "jax[cuda12]==0.6.2"
```

Restart the kernel after reinstalling if JAX/NumPy was already imported, then rerun the source cell and the GPU check. Do not reinstall only `jax` while leaving incompatible plugins installed. `jax`, `jaxlib`, `jax-cuda12-plugin` and `jax-cuda12-pjrt` must all be **0.6.2**, with no CUDA 13 plugin. The notebook's verification cell checks these versions and executes a compiled GPU operation before training.

Defaults are **4 eviction policies × 3 disk regimes × 3 seeds = 36 training runs**. There are 12 experimental configurations, each with three independently trained actors. Each SAC run also has two critics and their target copies. Extra networks do not themselves guarantee convergence. Every configuration gets the same architecture, budget and data splits; large-disk policies are mathematically redundant but remain separate controls.

Training uses 400 episodes, width 64, eight-observation histories, batch size 128, replay capacity approximately 30,000 decisions, and 0.25 gradient updates per new decision. Actor/critic learning rates are `1e-4`/`3e-4`, discount `.98`, target EMA `.005`. CPU or GPU can use the same commands and checkpoints. Start with a subset when estimating runtime:

```bash
python run.py train --data data --output outputs/trial --device cpu \
  --policies LRU --disks tight --seeds 0 --episodes 40 \
  --width 16 --history 4 --batch-size 16 --eval-every 4 --update-ratio .1
```

Select the same subset for `evaluate`. It requires the requested checkpoints and does not silently replace missing agents with heuristics. `--skip-existing` skips **completed** runs, which is useful for continuing the 36-run sweep. Use `--resume --episodes 400` with the original `--width` and `--history` to extend an interrupted or completed run to a total of 400 episodes. Weights, twin critics, targets, temperature and optimizer states are restored. Replay is rebuilt from `--warmup-episodes` on-policy training traces; behavior cloning is not repeated. Sampler state and trace order are restored when present in newer checkpoints; older checkpoints reset those deterministically. For checkpoints from the initial pilot/continued runs, add `--reward-mode completion`; `--reward-mode holding` is the unshaped causal ablation. Reward definitions cannot be changed on resume. This is a documented continuation, not a bit-for-bit replay-buffer resume. Existing CSVs and the best validation checkpoint are retained. Runs already at the requested episode target are skipped, so the same resume command can continue an interrupted seed sweep. Pickle checkpoints should come from trusted local experiments.

## Methodological decisions

The paper motivates heterogeneous compute, serial image pulls, CNN spatial features, GRU history, and SAC. Its main text specifies 20 nodes and 50 images, while the author config uses 15 nodes. The supplied code also references an undefined `RANDOM_SEED`, discards popped tasks when no node is feasible, produces same-task next observations, lacks eviction, and does not drain outstanding jobs before ending an episode. Slot-wide actions are selected before their memory/disk reservations are applied. We rebuild the simulator and learning loop rather than silently treating that implementation as a faithful reference.

The PDF's prose assumes cached images persist, while equation (14) reverses the monotonicity inequality. Supplementary appendices and an exact experimental config were not supplied. The numerical units also leave ambiguity between code-scale compute values and physical GHz/FLOPS. The following are explicit **reconstruction choices and extensions**. The corrected reward implementation is covered in the Learning and Potential shaping items below:

- **Topology and services.** Default: 14 CPU and six GPU nodes; 50 images of 3–15 GB. Every trace starts with empty node caches. Node CPU rates are 3–6.5 normalized compute units/s and GPU rates 10–16; CPU/GPU memory is 80–180/16–32 GB. Bandwidth is 2–6 Gbit/s. CPU service work is 1.5–2.5 units, GPU work 2–5. GPU services incur four times their work on CPU. CPU services cannot run on GPU nodes. These normalized compute units preserve hardware preference without claiming literal hardware calibration.
- **Concurrency.** Every arrival slot is 2 seconds. Tasks in a slot are admitted sequentially and immediately change state. Active computation shares node speed equally. Starts and completions trigger exact rate reallocation; GPU aggregate efficiency is multiplied by `(1-.05)^(active_count-1)`. GPU memory-transfer overhead is omitted. Memory is conservatively reserved from admission to completion, including image wait time. Task images remain pinned throughout that interval.
- **Transfers.** Each node serializes image downloads at `8 × image_GB / bandwidth_Gbit_per_s`; duplicate in-flight requests share one pull. Disk space is reserved before a pull starts. Uploads use independent distance-attenuated effective links `B/(1+(distance/35)^2)` and overlap image preparation; compute starts after both finish. This replaces unspecified radio constants and does not simulate shared wireless interference, cloud offload, registry congestion, or image layers.
- **Disk regimes.** Per-node capacity is `max(largest_image, fraction × total_catalogue_size)`. Fractions: **tight .045**, **medium .100**, **large 1.010**. Tight and medium were selected using a separate calibration trace and the Greedy_latency heuristic; this is workload calibration, not an unbiased held-out result. One 400-slot calibration trace gave 471/96/0 LRU evictions. Counts depend on the scheduler and trace. Large capacity guarantees no eviction even if every image is cached on every node. Fractions are stored in `world.json`; edit them there before starting a new experiment, or change `DISKS` before generating a new dataset.
- **Eviction.** Pulling and pinned images are ineligible. LRU minimizes last-access sequence; FIFO minimizes insertion sequence; LFU minimizes cache-resident reference count with LRU ties. GDSF uses `H = L + frequency × retrieval_cost / image_size`, raises per-node `L` to `max(L, evicted_key)` (pinned entries may retain older keys), and resets frequency upon reinsertion. Retrieval cost is `1 + 8 × size / bandwidth` seconds. The fixed registry cost prevents the size term cancelling completely. Eviction ties resolve deterministically. Reference counts include coalesced pull references. A hit means the image is **ready at admission**, not merely reserved.
- **Infeasibility.** When no compatible node has memory plus reclaimable disk capacity, record an explicit rejection. The rejection action is masked unless forced. Failure latency equals deadline allowance plus 50 seconds, yielding normalized reward `-1`. Tasks are never silently removed or retried. All admitted jobs drain after the last arrival. Rejections count as failures in completion/hit metrics and incur their explicit finite penalty in latency metrics.
- **Workloads.** Services persist and recur. Requests use a shuffled Zipf distribution (exponent 1.1), a 25% repeat-last-service component, a rotating hot set every 60 slots, bounded variable intensity, and 12% burst slots. Service work has ±20% jitter; deadlines are uniform 20–50 seconds after arrival. Deadlines are longer than the author's 3–5 slots because GB/Gbit conversion makes cold pulls much longer. Catalogue/topology are shared; request sequences are disjoint through `SeedSequence([catalogue_seed, split_tag, trace_index])`. Train/validation/test tags are 11/22/333; a separate calibration domain uses tag 33. Generalization here means new workloads on a known deployment and catalogue, not unseen services or topologies.
- **State.** Each node exposes resources, current-image readiness, estimated immediate latency, cache metadata for every image, current image identity and causal recent demand. Observation version 2 also exposes the known fraction of episode slots remaining; version-1 datasets/checkpoints remain supported. These are online features; the estimate excludes future arrivals. Full cache identities/ages/frequencies/priorities let the policy distinguish eviction consequences. Priorities/ages are normalized and bounded as features. Image sizes are fixed for a catalogue and may be learned through image identity.
- **Network.** Per-node projection → width-3 convolution across a stable CPU/GPU-and-x node ordering → mean/max pooling → GRU → shared per-node scoring head. Actor, twin critics and targets have separate parameters. Both critics also use history. This differs from the supplied flattened-feature CNN and actor-only GRU. GRU state is recomputed from the last eight observations, with padding at trace boundaries, rather than carried over an entire episode. Replay never crosses episode boundaries or stores stale hidden states.
- **Learning.** Masked discrete SAC sums exactly over valid actions. Both target and actor distributions use feasibility masks; forced rejection keeps masks nonempty. The twin-critic loss is Huber, gradients are clipped, and temperature targets `.65 × log(number_of_valid_actions)`. The base reward is `(current_task_deadline_allowance - interval_latency_cost)/50`. Interval cost integrates the number of admitted, unfinished tasks over elapsed simulator time, plus an explicit rejection cost. The final transition includes the complete drain. Its undiscounted episode sum exactly equals total deadline slack / 50, but every reward depends only on the current decision and the following interval. The next state is the next scheduling decision.
- **Potential shaping (default).** Sparse interval costs are shaped by `.98*Phi(next_state)-Phi(current_state)`, with `Phi=-predicted_remaining_holding_cost/50`. The prediction privately drains only already admitted work, assuming no future arrivals; it reads no future requests and does not mutate the simulator. Initial/terminal potentials are zero. This telescopes out of the discounted return, preserving the base discounted objective while providing immediate credit for queue growth and externalities. `--reward-mode holding` disables shaping. This is an explicit model-assisted training choice; no projection or heuristic is used by the learned actor at evaluation. `--reward-mode completion` retains the initial, retrospective per-task reward only for reproducibility; those rewards can depend on later actions and are not the current default. Replay is uniform over retained decisions and stores observations in float16, reconstructed as float32 batches. Whole episodes are evicted, so one episode may exceed the nominal capacity.
- **Warm start.** Two training traces are scheduled by Greedy_latency; the actor gets 100 supervised behavior-cloning updates before SAC. This is disclosed assistance, not a paper-faithful baseline. Set `--bc-updates 0` for the pure-RL ablation; set `--warmup-episodes 0` too to remove heuristic replay. Episode zero is logged **after** warm start. There is no heuristic fallback hidden inside the agent at evaluation.
- **Local-feature ablation.** `--local-branch` adds a zero-initialized, fully learned affine score from each node's first 18 immediate features alongside its CNN–GRU score. This preserves the identity of local resources/readiness while the convolution mixes neighbouring nodes. It applies to both actor and critics, has no fixed latency weights or heuristic fallback, and is disabled by default. Checkpoints record whether it is present; older checkpoints load without it. Resuming retains the saved architecture.
- **Baselines.** Greedy_latency includes upload, image waiting and current compute contention; Image_priority prefers ready images then estimated latency; Least_load counts admitted unfinished jobs; Greedy_resources uses speed × remaining-memory fraction; Random samples feasible nodes. All share physics, masks, traces and rejection handling. Other paper DRL baselines (DQN/PPO/DDPG/GNN) are outside this focused extension.

The simulator is a conservative research abstraction. Synthetic traces, normalized compute, fixed bandwidth and small numbers of training seeds limit real-system conclusions. Deadline success is measured, not guaranteed by admission control. Changing disk size can change feasible actions and placements, so even a large disk need not improve an untrained agent.

## Convergence and results

Validation traces are fixed and never enter replay. Select `best.pkl` by mean validation deadline slack, then evaluate that frozen checkpoint on test traces. `last.pkl` preserves the final iterate. Evaluation writes unrounded per-trace/per-seed metrics to `results.csv`. `training.csv` contains actual rollout metrics (its `reward` column is mean realized deadline slack / 50) plus losses/entropy/temperature/gradient norms; `signals.csv` separately records the actual learning-reward mean and discounted learning return; `validation.csv` includes episode zero and regular deterministic evaluations. Source settings, device/version information, catalogue SHA-256 and per-run completion records accompany the checkpoints.

`plot` generates:

- `figures/convergence.png` and `.pdf`: all requested policy/disk configurations, individual seed curves, seed mean ± one standard deviation, and a **validation** Greedy_latency reference. No smoothing or fabricated interpolation is applied to recorded points.
- `figures/results.png` and `.pdf`: mean latency, on-time fraction, cache hit fraction and eviction counts for every method/policy/disk setting.
- `figures/diagnostics.png` and `.pdf`: critic loss, entropy and temperature at the tightest requested disk.
- `summary.json`: 95% percentile intervals from crossed resampling of training seeds and shared test traces; heuristics have one deterministic copy per trace. One seed or one trace cannot estimate its missing variance component.
- `paired_comparisons.json`: agent-minus-Greedy_latency test latency with the trace pairing retained. Negative differences favor the agent.
- `convergence_diagnostics.json`: initial/final validation reward, final-quarter slope, number of observations and seeds. Short smoke runs explicitly report insufficient evidence. A flat curve alone is not a success criterion: compare validation return, seed variability, success rate and the greedy reference.

The full 36-run GPU sweep remains to be run on a GPU. CPU smoke results validate mechanics only. The initial `outputs/pilot`, `outputs/continued`, and `outputs/local_ablation` used the legacy completion reward and are retained as diagnostic experiments; they do not validate the corrected causal trainer. Current smoke artifacts are under `outputs/smoke_current`; `outputs/causal` and `outputs/potential` compare causal rewards without/with potential shaping. The current three-seed CPU run uses shorter traces and a width-16 network; its exact commands and measured results are recorded in `outputs/potential/CPU_VALIDATION.md`. Those curves are evidence of learning, not proof of the paper's gains or convergence in all 12 configurations. Historical pilot reports are marked explicitly.

## Replay and animation

Evaluation records `outputs/<run>/replays/<method>_<policy>_<disk>_<trace>.json`, default first 200 slots (`--replay_slots N`). A log contains static node positions/types/capacities, image sizes, cache snapshots per slot, and per-task origin/type/image/node, hit/coalesced/miss, victim records, pull start/finish, upload completion, actual finish and deadline outcome. Cache snapshots reference image IDs; pin state and download progress are reconstructed from the recorded events. Recording does not change simulator decisions.

```bash
python viz.py --replay outputs/full/replays/Agent_LRU_tight_test_0000.json --mode html --fps 10
python viz.py --replay outputs/full/replays/Agent_LRU_tight_test_0000.json --mode mp4 --fps 10
# Two or four paths compare panels in lockstep; all must have the same trace.
python viz.py --replay outputs/full/replays/scenario_four_policies.json --mode html
```

`viz.py` imports no simulator, JAX, torch or GPU code. HTML mode also avoids NumPy and matplotlib. Exported HTML embeds JSON, CSS and canvas JavaScript, including play/pause, speed control, a seek slider and responsive layouts. CPU nodes are squares, GPU nodes hexagons; task shapes distinguish preference; image colors are consistent across panels. Disk blocks show reserved size and pull progress with white pin outlines, red eviction fades and victim labels. Late tasks turn red. Live metrics use only outcomes completed by the current replay time; on-time/arrived consequently differs from the final eventual-success metric until all work finishes. Simulated transfer travel is slowed visually; timestamps are not changed. Playback shows .75 screen seconds per simulator slot at 1× speed; FPS controls video sampling.

`python viz.py --scenarios --root outputs/full` emits `replay_agent_vs_greedy`, `replay_four_policies`, and `replay_disk_pressure`, each as `.html` and `.mp4`/`.gif`. The four-policy and disk comparisons use the **same actor trained on LRU/tight**, not four separately trained policies. Benchmark results use the separately trained policies. These controlled comparisons test intervention on a frozen actor and may include distribution shift. `--smoke --root outputs/smoke` renders the first 16 slots of all three scenarios.

To produce controlled scenarios from a single trained checkpoint:

```bash
python run.py replay --data data --output outputs/showcase \
  --checkpoint outputs/full/models/LRU_tight_s0/best.pkl --replay_slots 200
python viz.py --scenarios --root outputs/showcase --fps 10
```

## References

The local PDF is the primary CHES source. Discrete SAC follows [Christodoulou (2019)](https://arxiv.org/abs/1910.07207). The GDSF key and resident-frequency update are described in [Yang et al., *A Data Cube Model for Prediction-Based Web Prefetching*, §4.2](https://cse.hkust.edu.hk/~qyang/Docs/2002/datacube.pdf); our retrieval-cost choice is stated above. JAX installation details come from the [official JAX documentation](https://docs.jax.dev/en/latest/installation.html).

Potential shaping follows [Ng, Harada and Russell (1999)](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf). Tests check the discounted-return telescoping identity and projected-drain equality directly.
