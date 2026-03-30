# Repository Guidelines

## Project Structure

- **`libriichi/`** — Rust high-performance Mahjong engine (PyO3 extension). Contains game rules, state machine, feature extraction, and inline tests. Exposes 6 PyO3 sub-modules: `consts`, `state`, `dataset`, `arena`, `stat`, `mjai`.
- **`mortal/`** — PyTorch training pipeline. Model definitions (`model.py`), supervised / oracle-dropout / online training scripts, A/B runners, evaluation, and all configs.
- **`exe-wrapper/`** — Small Rust helper binary crate (workspace member).
- **`scripts/`** — Windows `.bat` entry points for build and training.
- **`checkpoints/`** — Model weight outputs (not committed).

## Documentation Layout

- **`docs/agent/current-plan.md`** — default handoff entry for new agent sessions; only current mainline, current live status, and immediate next step.
- **`docs/agent/mainline.md`** — stable defaults, stage map, frozen conclusions, and current active run.
- **`docs/agent/experiment-workflow.md`** — current stage workflow, manual stop points, and operating discipline.
- **`docs/agent/laptop-remote-ops.md`** — laptop-node remote-operation notes: SSH / PowerShell behavior, dataset status, current loader defaults, and proven remote execution patterns.
- **`docs/agent/code-sync.md`** — default desktop↔laptop code-sync scheme: current canonical branch (`main`), desktop as source of truth, laptop bare mirror layout, and the one-command sync path.
- **`docs/status/stage05-verified-status.md`** — manually maintained verified Stage 0.5 status; use this when the auto-generated summary lags behind raw artifacts.
- **`docs/status/p1-selection-canonical.md`** — the only valid P1 winner-selection rubric.
- **`docs/status/stage05-fidelity-results.md`** — auto-generated Stage 0.5 fidelity run snapshot. It is run-scoped, not the default handoff; if it conflicts with `current-plan.md`, `stage05-verified-status.md`, or `p1-selection-canonical.md`, prefer those docs.
- **`docs/research/`** — human-oriented methodology, engineering, and experiment notes; not the default current-state handoff.
- **`docs/reflections/`** — personal reflection and human-AI collaboration notes; background only.
- **`docs/archive/`** — historical snapshots and retired long-form docs; never treat archive docs as current defaults unless a current entry doc explicitly revives them.
- **`README.md` / `docs/README.md`** — quickstart plus the human-oriented documentation index.

## Build, Test, and Development Commands

```powershell
# Environment (name in environment.yml is "mortal"; scripts may say "mahjong")
conda env create -f environment.yml
conda activate mortal

# Build Rust engine → installs as importable Python package
.\scripts\build_libriichi.bat          # runs: maturin develop --release --manifest-path Cargo.toml

# Verify build
python -c "import libriichi; print('OK')"

# Rust tests
cargo test -p libriichi                # all engine tests
cargo test -p libriichi state::test    # targeted state module tests

# Rust tests from a plain PowerShell shell may pick an unsupported global Python.
# If that happens, pin PyO3 to the mortal env and prepend its DLL dirs to PATH:
$env:PYO3_PYTHON="C:\ProgramData\anaconda3\envs\mortal\python.exe"
$env:PATH="C:\ProgramData\anaconda3\envs\mortal;C:\ProgramData\anaconda3\envs\mortal\Library\bin;C:\ProgramData\anaconda3\envs\mortal\Scripts;$env:PATH"
cargo test -p libriichi state::test

# Python smoke test
python mortal\test_greedy.py

# Current training entry points
.\scripts\run_grp.bat                  # Stage 0: cd mortal && python train_grp.py
.\scripts\run_supervised.bat           # Stage 0.5: formal supervised pretraining / protocol replay
.\scripts\run_stage1_refine.bat        # Stage 1: oracle-dropout supervised refinement
.\scripts\run_stage1_ab.bat            # Stage 1 Block C: recipe / gamma A-B runner
.\scripts\run_online.bat               # Stage 2: cd mortal && python train_online.py

# Formatting
cargo fmt                              # Rust formatting (always run before commit)
```

## Code Style

### Rust (`libriichi/`)
- **Edition 2024**, workspace resolver 3, release profile: `lto = true, codegen-units = 1`.
- `lib.rs` enforces ~75 strict clippy lints via `#![deny(...)]` — including `float_cmp`, `undocumented_unsafe_blocks`, `use_self`, `uninlined_format_args`, `get_unwrap`, `string_add`, `trivially_copy_pass_by_ref`, and many more. One `#![allow(clippy::manual_range_patterns)]` for the `matches_tu8` macro.
- snake_case modules, inline `#[cfg(test)] mod test` blocks. Tests use JSON mjai events to drive `PlayerState` updates (see `state/test.rs`).
- Uses `mimalloc` global allocator (default feature).

### Python (`mortal/`)
- 4-space indent, snake_case functions/variables, PascalCase classes.
- `config.py` is 4 lines: loads `mortal/config.toml` (or `$MORTAL_CFG`), no validation. All type safety is the caller's responsibility.
- `prelude.py` configures logging (INFO to stderr), silences warnings, sets UTF-8 stdin.
- GRP precision is now configurable; for this machine and the current strongest GRP setup, prefer `torch.float32`.

## Architecture — Critical Constants

These are defined in `libriichi/src/consts.rs` and imported in Python via `from libriichi.consts import ...`:

| Constant | Value | Notes |
|----------|-------|-------|
| `ACTION_SPACE` | 46 | 37 discard + 1 riichi + 3 chi + 1 pon + 1 kan + 1 agari + 1 ryukyoku + 1 pass |
| `obs_shape(v4)` | (1012, 34) | Normal observation channels — defined in `state/obs_repr.rs` |
| `oracle_obs_shape(v4)` | (217, 34) | Perfect-info Oracle channels — defined in `dataset/invisible.rs` |
| `GRP_SIZE` | 7 | GRP input dimension |
| `MAX_VERSION` | 4 | Current feature version |

**Do not modify feature channel counts** without updating both Rust extraction code and Python model input dimensions.

## Architecture — Neural Network (`mortal/model.py`)

| Component | Key Details |
|-----------|-------------|
| **Brain (Encoder)** | 1D-ResNet, 40 blocks × 192 channels (config: `[resnet]`), GroupNorm(32), Mish activation, pre-activation ResBlocks with SE-style `ChannelAttention(ratio=16)` in every block. Output: 1024-dim. |
| **CategoricalPolicy** | Linear(1024→256) + tanh + Linear(256→46), orthogonal init |
| **DQN** | Linear(1024→47), Dueling split: V(1) + A(46) |
| **AuxNet** | Linear(1024→sum(dims)), bias=False — ranking prediction head |
| **GRP** | GRU(7, hidden=384, 3 layers, float32) → FC(1152→1152→24), 24 = 4! ranking permutations |

## Architecture — Current Training Pipeline

1. **Stage 0 — GRP** (`train_grp.py`): Trains Global Reward Predictor on game logs. Output: `checkpoints/grp.pth`.
2. **Stage 0.5 — Supervised Pretraining / Protocol Search** (`train_supervised.py`, `run_stage05_formal.py`, `run_stage05_fidelity.py`): Selects the strongest supervised protocol under temporal drift and saves `best_loss / best_acc / best_rank`.
3. **Stage 1 — Oracle Dropout Supervised Refinement** (`train_stage1_refine.py`): Current default mainline direction. Continues from Stage `0.5` top-k protocol seeds with `policy CE + rank aux + opponent_state aux + danger aux`.
4. **Stage 2 — PPO Online** (`train_online.py`): Self-play with dynamic entropy regularization. Run after the new Stage `1` line stabilizes.

## Configuration (`mortal/config.toml`)

All hyperparameters centralized here. Key sections: `[control]`, `[resnet]`, `[policy]`, `[oracle]`, `[aux]`, `[optim]`, `[dataset]`, `[grp]`, `[online]`, `[env]`, `[1v3]`.
- Use `mortal/config.example.toml` as template for new environments.
- `[dataset]` paths (e.g., `D:/mahjong_data/...`) are local — never commit real paths.
- `MORTAL_CFG` env var overrides the config file path.

## Rust ↔ Python Integration

| Pattern | Details |
|---------|---------|
| **Import** | `from libriichi.consts import obs_shape, ACTION_SPACE`; `from libriichi.dataset import GameplayLoader, Grp`; `from libriichi.state import PlayerState` |
| **Data flow** | `GameplayLoader` (Rust) parses `.json.gz` → Python `FileDatasetsIter(IterableDataset)` → PyTorch DataLoader |
| **Engine** | `mortal/engine.py` wraps model inference: `MortalEngine.react_batch()` takes numpy obs/masks, returns action lists |
| **NumPy bridge** | Rust uses `numpy::PyArray1/2`; Python bridges via `np.stack` + `torch.as_tensor` |

## Testing Guidelines

- Rust: add `#[test]` in the same module file (e.g., `state/test.rs` contains ~1400 lines of inline tests driven by JSON mjai events).
- Python: add `test_*.py` near related code.
- Run targeted tests first: `cargo test -p libriichi state::test` or a focused Python script.
- No coverage gate — but new behavior needs at least one reproducible test.

## Local Hardware Profile

- **CPU**: Intel Core i5-13600KF
- **GPU**: NVIDIA GeForce RTX 5070 Ti
- When suggesting training settings or performance tweaks, assume this machine as the default target.
- Prefer recommendations that balance DataLoader throughput, CPU preprocessing, disk I/O, and GPU utilization for this hardware pair.
- For the current `384x3` `fp32` GRP training setup on this machine, treat `num_workers = 10` as the practical default and only move lower or higher with task-specific evidence.

## Multi-Machine Compute Topology

- **Primary desktop**: Intel Core i5-13600KF + NVIDIA GeForce RTX 5070 Ti. This remains the default target when a note only says "this machine".
- **Secondary laptop node**: Intel Core i9-13900HX + NVIDIA GeForce RTX 4060 Laptop GPU (`8 GB` VRAM) + `32 GB` DDR5. Treat it as an additional independent experiment runner, not as an already-wired distributed training worker.
- Use the laptop for parallel GRP runs, Stage `0.5` loader / validation benchmarking, supervised A/Bs, and shorter Stage `1` probes when the desktop is busy. Do not assume cross-machine gradient sync, shared replay buffers, or checkpoint co-writing unless that plumbing is explicitly added for the task.
- Canonical development branch is now local `main`, which tracks `origin/main`.
- Source-of-truth code lives in the desktop `main` worktree first; sync to the laptop through the documented bare-mirror workflow in `docs/agent/code-sync.md`, then push to GitHub `origin/main` as needed.
- Do not trust an older copied workspace on the laptop without an explicit resync.
- Laptop repo default path: `C:\Users\numbe\Desktop\MahjongAI`
- Laptop Conda env: `C:\Users\numbe\miniconda3\envs\mortal`
- Desktop-to-laptop shell access is available over LAN SSH via the desktop key `C:\Users\numbe\.ssh\mahjong_laptop_ed25519`. The laptop LAN IP can change, so re-check it before hardcoding commands.
- When running the same stage on both machines, always use distinct run names / output directories tagged by machine, and never let both machines write to the same checkpoint path or log directory.
- Current laptop Stage `0.5` loader default from the `2026-03-30` local-subset benchmark:
  - train: `num_workers = 6`, `file_batch_size = 7`, `prefetch_factor = 3`
  - val: `val_file_batch_size = 7`, `val_prefetch_factor = 6`
  - close validation fallback: `7 / 5`
- Benchmark artifacts copied back to the desktop repo:
  - `logs/laptop_stage05_loader_bench/summary.json`
  - `logs/laptop_stage05_loader_bench/confirm_summary.json`
- Important scope note: the laptop benchmark used a copied representative subset because the full dataset root is not yet mirrored onto the laptop. Treat these as the current operational defaults for the laptop, but still prefer a full-data recheck before locking them in as a permanent global default.

## User Objective

- The primary objective is to build the strongest Mahjong AI possible on this machine, not merely the fastest or cheapest-to-train model.
- When proposing architecture, training, or data-pipeline changes, optimize for final playing strength first and throughput second, as long as the setup remains practical on the local hardware profile above.
- For auxiliary models such as GRP, evaluate trade-offs by likely downstream impact on Stage 1/2 policy quality, not just standalone validation speed.
- When two options are close in expected final strength, prefer the smaller or faster option; when gains are meaningful, prefer the stronger option even if training is slower.
- Current GRP guidance from local benchmarking: for this machine, the strongest practical final GRP setup is currently `384x3` trained in `fp32` with validation-loss-driven checkpointing and LR scheduling. If prioritizing efficiency over peak strength, `256x3` is the best practical fallback. Depth `x4`, `fp64`, and very large widths currently show diminishing returns relative to the extra cost.
- Stage `0.5` / supervised validation default: use validation-only `val_file_batch_size = 8` and `val_prefetch_factor = 5` on this machine. If validation hits a loader/resource error, keep the same validation settings and retry indefinitely; do not downgrade to a safe single-process validation mode. The validated fix path is to explicitly close validation iterators/workers after each pass and release the training loader before budget-bound validation, because the previous `1455` issue was shared-mapping lifetime pressure rather than simple RAM exhaustion.
- Stage `0.5` / supervised training default: keep heavy action/scenario selection metrics out of the per-batch training hot path. Training should keep only lightweight optimization/basic monitoring metrics; full discard/decision/sliced-scenario metrics belong to validation unless a run explicitly asks otherwise.
- Stage `0.5` / supervised validation memory default: training stays at `4/10/3`; validation stays at `8/5`. Treat `8/5` as the default long-run operating point on this machine unless a fresh benchmark shows a stronger speed/stability trade-off.

## Local Python Environment

- **Preferred environment**: `conda activate mortal`
- **Verified interpreter**: `C:\ProgramData\anaconda3\envs\mortal\python.exe` (`Python 3.12.12`)
- **Verified `libriichi` install**: `C:\Users\numbe\AppData\Roaming\Python\Python312\site-packages\libriichi`
- **Verified extension file**: `C:\Users\numbe\AppData\Roaming\Python\Python312\site-packages\libriichi\libriichi.cp312-win_amd64.pyd`
- The current plain-shell default `python` is `C:\Python314\python.exe` (`Python 3.14.3`). PyO3 `0.23.4` does **not** support Python 3.14, so a bare `cargo test` can fail during PyO3 build discovery unless `PYO3_PYTHON` is pinned to the `mortal` interpreter above.
- On this machine, `conda` is not guaranteed to be on `PATH` in a fresh PowerShell session. Prefer the absolute interpreter path `C:\ProgramData\anaconda3\envs\mortal\python.exe` when you only need Python, instead of assuming `conda run -n mortal ...` will work.
- Rust test binaries may also fail with `STATUS_DLL_NOT_FOUND` unless the `mortal` env directories are prepended to `PATH` before running `cargo test`:
  `C:\ProgramData\anaconda3\envs\mortal`
  `C:\ProgramData\anaconda3\envs\mortal\Library\bin`
  `C:\ProgramData\anaconda3\envs\mortal\Scripts`
- When running training, smoke tests, import checks, or Rust tests that touch PyO3, prefer the `mortal` environment or the absolute `mortal` interpreter plus the PATH setup above.

## Project Conventions (Non-obvious)

- **Conda env name mismatch**: `environment.yml` says `mortal`, batch scripts activate `mahjong`. Be aware when writing scripts.
- **CPU affinity is now opt-in**: training entry points no longer default to `p_cores`. Leave `MORTAL_CPU_AFFINITY` unset for normal Windows scheduling, or set it explicitly to values such as `p_cores`, `all`, or a CPU list/mask when you really want pinning.
- **GroupNorm default**: `Brain.__init__` defaults to `"BN"`, but actual config and `player.py` both use `Norm="GN"` (GroupNorm, 32 groups). Always pass norm explicitly.
- **SE Attention in every ResBlock**: Unlike standard ResNets, every block has channel attention — do not remove it.
- **Pre-activation order**: V3/V4 use pre-activation ResBlocks (Norm→Activ→Conv).
- **`common.py` TCP**: `drain()`/`submit_param()` use raw TCP sockets with `torch.save/load` serialization. `recv_msg()` currently uses `weights_only=False` (known TODO).

## Deployment & Online Self-Play

There is no formal CI/CD pipeline. The project runs locally as a research training framework on Windows.

### Online Self-Play Architecture (Stage 2)
- **`server.py`** — `ThreadingTCPServer` at `127.0.0.1:5000`. Manages param distribution and replay buffer. Trainers call `drain` to pull replays and `submit_param` to push new weights.
- **`client.py`** — Worker that polls the server for latest params, runs self-play games via `TrainPlayer.train_play()`, and submits replays back.
- **`train_online.py`** — Trainer loop: drains replays from server, runs PPO updates, pushes new params.
- Buffer/drain directories configured in `config.toml [online.server]` with `buffer_dir`, `drain_dir`, `capacity`.

### Evaluation
- **`one_vs_three.py`** — Runs challenger vs champion (or vs akochan) matches using `libriichi.arena.OneVsThree`. Configured via `config.toml [1v3]`.
- **TensorBoard**: `tensorboard --logdir ./mortal/tb_log_supervised_main` (Stage `0.5`), `./mortal/tb_log` (legacy Stage `1` / Stage `2`), or `./mortal/tb_log_grp` (Stage `0`).

### Checkpoint Artifacts
- `checkpoints/grp.pth` — Stage 0 output (GRP weights).
- `checkpoints/grp_latest.pth` — latest resumable GRP training state; use for continuing Stage 0 training, not as the default downstream model.
- `checkpoints/grp_best_acc.pth` — GRP checkpoint with the best validation exact-permutation accuracy; keep as a secondary candidate for downstream A/B checks.
- `checkpoints/stage0_5_supervised*.pth` — Stage `0.5` formal checkpoints (`best_loss / best_acc / best_rank / latest`).
- `checkpoints/online_ppo/` — Stage 2 periodic saves.
- State files contain `{'mortal': ..., 'policy_net': ..., 'config': ...}` dicts loaded via `torch.load(..., weights_only=True)`.

### GRP Checkpoint Policy
- Treat GRP checkpoints as three roles: `best_loss` for default downstream use, `best_acc` as a backup candidate, and `latest` only for resume.
- Default Stage 1/2 training should load the `best_loss` GRP checkpoint unless there is explicit evidence that `best_acc` produces a stronger downstream policy.
- Use `best_acc` only for controlled comparisons; do not silently replace `best_loss` with it in the main pipeline.
- When changing GRP architecture or dtype, prefer starting a fresh Stage 0 run rather than resuming from an incompatible `latest` state.

## Security Notes

- `common.py` TCP communication (`127.0.0.1:5000`) has no TLS or authentication.
- `recv_msg()` uses pickle deserialization (`weights_only=False`) — do not expose to untrusted input.
- Release profile has `overflow-checks = false`.
- Do not commit `config.toml` with real dataset paths or credentials.

## Commit Guidelines

- Short, imperative, scoped subjects: `mortal: fix oracle dropout schedule`, `libriichi: add chi validation test`.
- PRs should list which stage/crate changed, config/path updates, and include training metric screenshots when applicable.
