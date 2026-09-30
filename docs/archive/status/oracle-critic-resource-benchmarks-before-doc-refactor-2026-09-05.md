# Oracle Critic Resource Benchmarks

> 历史归档 · 归档整理：2026-09-05。正文保留当时的事实、判断和命令，不作为当前运行依据。当前入口见 [文档地图](../../README.md)；旧 SL / RL 强度及 Oracle 验证结论须结合 [独立审计](../../research/sl-rl-audit-2026-09-05.md) 阅读。

Date: 2026-05-23

Scope: local desktop Oracle critic pretraining probes for the dual-tower `all`
train scope. Each probe used `mortal.online.pretrain_oracle_critic` through
`scripts/probe_oracle_critic_resources.py`, with short windows intended to find
a safe high-throughput operating point before longer experiments.

Online C/D/E resource notes are included here as run-safety defaults, even
though the pretraining table below is the main scope.

## Current Recommendation

Use this as the initial dual-tower short-probe default on the desktop:

```toml
[oracle_critic_pretrain]
critic_arch = "dual_tower"
train_scope = "all"
batch_size = 640
num_workers = 3
file_batch_size = 6
prefetch_factor = 2
val_num_workers = 0
val_file_batch_size = 8
val_prefetch_factor = 5
```

Rationale: `dual_b640_w3_f6_p2_s500` completed cleanly at about `3.10`
steps/sec, with peak GPU memory around `11.3 GB`, peak system memory around
`22.1 GB` (`~69.5%`), and GPU temperature peaking at `71C`. This is currently
the best balance observed for short resource probes.

Use this as the safer full-file-pool long-run default:

```toml
[oracle_critic_pretrain]
critic_arch = "dual_tower"
train_scope = "all"
batch_size = 640
num_workers = 2
file_batch_size = 6
prefetch_factor = 2
val_num_workers = 0
val_file_batch_size = 8
val_prefetch_factor = 5
```

Rationale: `dual_b640_w2_f6_p2_full_s30000` completed a real full-pool 30k run
without memory pressure: GPU utilization median `79%` / max `92%`, GPU memory
max `11.1 GB`, system RAM median `67.4%` / max `74.0%`, and GPU temperature max
`75C`. Use its `checkpoints/best.pth` for dual-tower D/E initialization.

Fallback if other work is also using RAM or the run becomes unstable:

```toml
[oracle_critic_pretrain]
batch_size = 512
num_workers = 2
file_batch_size = 6
prefetch_factor = 2
```

`dual_b512_w2_f6_p2_s180` completed cleanly with peak GPU memory around
`9.4 GB` and peak system memory around `19.7 GB`.

For dual-tower online C/D/E runs, use batch `192` as the safe default on the
desktop. For controlled-seed runs, set `allow_cudnn_benchmark=true`; otherwise
`repro.enabled=true` disables cuDNN benchmark and can pick a much higher-memory
convolution path. Batch `224`/`288`/`320` are useful manual short-window or
resume acceleration points, but they are not safe defaults under the current
guard.
Batch `320`
reached step `48000` in `dual_E_w3000_s50000_b320_resume20000` and then
correctly stopped on `gpu_mem_mb>=15000.0x3`; batch `288` finished that 50k run
but later stopped a 3000-step D sanity at step `2500` on
`system_mem_percent>=85.0x3`. Batch `224` then resumed the same D run from
`2500` to `3000` with system RAM max `80.5%` and GPU memory max `10.5 GB`, but
the longer `dual_D_s20000_fix_old_policy_entropy_b224_resume3000` window later
showed a role-level system RAM spike of `87.97%` around step `5000`.
After fixing the old-policy update to reuse modules instead of `deepcopy` on
GPU, `dual_D_s5000_value_w002_cudnnbench_b192_resume3000` finished cleanly:
GPU memory max `9.4 GB`, system RAM max `83.85%`, and about `4.25` steps/sec.
The longer `dual_D_s20000_value_w002_cache16_cudnnbench_b192_resume5000`
completed from the same 5k checkpoint with cache16 replay IS enabled: GPU
memory max `10.64 GB`, system RAM max `90.59%`, and about `2.63` steps/sec.
That is close to the desktop RAM ceiling but still below the `92%` hard guard.
The same family of probes without `allow_cudnn_benchmark=true` hit
`gpu_mem_mb>=15000.0x3`, so that flag is part of the online resource default,
not a cosmetic setting.

The follow-up `dual_E_w3000_s50000_1v3` small gate used a single `1v3` shard
with `2000` games. It was resource-light at about `4.1 GB` GPU memory during
polling, so it does not affect online training batch defaults.

For full-file-pool pretraining, prefer `num_workers=2` for the first 30k run.
`dual_b640_w3_f6_p2_full_s3000` learned cleanly, but system RAM peaked at
`79.5%`, which is too close to the long-run comfort boundary.

## Probe Summary

| Run | Steps | Batch | Workers | File Batch | Prefetch | Steps/sec | GPU Mem Max | Sys Mem Max | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `dual_b512_w4_f10_p3_s120_r2` | 120 | 512 | 4 | 10 | 3 | n/a | 9.9 GB | 30.5 GB | Too close to RAM limit |
| `dual_b384_w2_f6_p2_s180` | 180 | 384 | 2 | 6 | 2 | n/a | 7.6 GB | 18.9 GB | Safe but underuses GPU |
| `dual_b512_w2_f6_p2_s180` | 180 | 512 | 2 | 6 | 2 | n/a | 9.4 GB | 19.7 GB | Safe fallback |
| `dual_b640_w2_f6_p2_s500` | 500 | 640 | 2 | 6 | 2 | 3.01 | 10.9 GB | 19.3 GB | Safe |
| `dual_b640_w3_f6_p2_s500` | 500 | 640 | 3 | 6 | 2 | 3.10 | 11.3 GB | 22.1 GB | Recommended |
| `dual_b640_w3_f6_p2_s3000` | 3000 | 640 | 3 | 6 | 2 | 3.64 | 10.7 GB | 24.3 GB | Passed 3000-step sanity |
| `dual_b640_w3_f6_p2_full_s3000` | 3000 | 640 | 3 | 6 | 2 | 3.97 | 10.7 GB | 25.3 GB | Full-pool sanity passed; RAM near limit |
| `dual_b640_w2_f6_p2_full_s30000` | 30000 | 640 | 2 | 6 | 2 | 3.75 | 11.1 GB | 23.5 GB | Completed full-pool 30k; use best for D/E |
| `dual_b640_w4_f6_p2_s300` | 300 | 640 | 4 | 6 | 2 | 2.82 | 10.8 GB | 24.1 GB | More RAM, slower |
| `dual_b768_w2_f6_p2_s120` | 120 | 768 | 2 | 6 | 2 | 1.99 | 12.3 GB | 20.2 GB | Slower; not worth it |
| `dual_D_s3000_fix_old_policy_entropy` | 2500 | 288 | n/a | n/a | n/a | 1.50 | 13.3 GB | 90.3% | Stopped by system RAM guard |
| `dual_D_s3000_fix_old_policy_entropy_b224_resume2500` | 3000 | 224 | n/a | n/a | n/a | 6.12 from resume | 10.5 GB | 80.5% | Safe for short resume |
| `dual_D_s20000_fix_old_policy_entropy_b224_resume3000` | 5000 | 224 | n/a | n/a | n/a | n/a | 10.6 GB | 88.0% role spike | Too tight for long default |
| `dual_D_s5000_value_w002_cudnnbench_b192_resume3000` | 5000 | 192 | n/a | n/a | n/a | 4.25 | 9.4 GB | 83.85% | Safe online 5k gate |
| `dual_D_s20000_value_w002_cache16_cudnnbench_b192_resume5000` | 20000 | 192 | n/a | n/a | n/a | 2.63 | 10.64 GB | 90.59% | Completed 20k; near RAM ceiling |
| `dual_D_s20000_value_w002_vtrace_r1c1_cache16_cudnnbench_b192_resume8000_retryconn` | 20000 | 192 | n/a | n/a | n/a | 3.10 | 12.31 GB | 89.94% | Completed V-trace 20k; resource-safe but weak 1v3 |
| `dual_E_w3000_s12000_value_w002_cache16_cudnnbench_b192_resume5000` | 12000 | 192 | n/a | n/a | n/a | 3.40 | 10.63 GB | 90.42% | Completed E 12k; near RAM ceiling |
| `dual_E_w3000_s8000_value_w002_cache16_cudnnbench_b192_resume5000` | 8000 | 192 | n/a | n/a | n/a | 5.06 | 9.65 GB | 90.76% | Completed freeze-only 8k; same 1v3 as 5k winner |
| `dual_E_w0_s9000_value_w002_cache16_cudnnbench_b192_resume8000` | 9000 | 192 | n/a | n/a | n/a | 11.54 | 11.16 GB | 82.63% | Completed 1k unfreeze probe; actor drift in 1v3 |
| `dual_E_w0_s9000_ef047_er1e3_value_w002_cache16_cudnnbench_b192_resume8000` | 9000 | 192 | n/a | n/a | n/a | 11.77 | 9.95 GB | 82.58% | Completed entropy-floor 1k unfreeze probe; no 1v3 improvement |
| `dual_E_w0_s9000_clip010_value_w002_cache16_cudnnbench_b192_resume8000_retry` | 9000 | 192 | n/a | n/a | n/a | 11.68 | 10.67 GB | 83.51% | Completed clip 0.1 1k unfreeze probe; current best 9k 1v3 |
| `dual_E_w0_s12000_clip010_value_w002_cache16_cudnnbench_b192_resume11500` | 12000 | 192 | n/a | n/a | n/a | 27.40 | 10.92 GB | 80.29% | Completed 11.5k->12k repair; 1v3 regressed to -1.7775 |
| `dual_E_w0_s10000_clip010_actorlr05_value_w002_cache16_cudnnbench_b192_resume9000_retry2` | 10000 | 192 | n/a | n/a | n/a | 14.85 | 11.49 GB | 84.50% | Completed actor LR scale 0.5 probe; 1v3 regressed to -3.69 |
| `dual_E_w0_s10000_clip010_policyheadlr05_value_w002_cache16_cudnnbench_b192_resume9000` | 10000 | 192 | n/a | n/a | n/a | 14.85 | 11.36 GB | 84.37% | Completed policy-head LR scale 0.5 probe; 1v3 regressed to -2.79 |
| `dual_E_w0_s10000_clip010_policyupd2_value_w002_cache16_cudnnbench_b192_resume9000` | 10000 | 192 | n/a | n/a | n/a | 14.73 | 11.03 GB | 82.99% | Completed policy update interval=2 probe; 1v3 regressed to -1.9575 |
| `dual_E_w0_s10000_clip010_policystop_value_w002_cache16_cudnnbench_b192_resume9000` | 10000 | 192 | n/a | n/a | n/a | 15.28 | 10.94 GB | 84.69% | Completed value-only continuation; 1v3 +0.8775 |
| `dual_C_w3000_s5000_value_w002_clip010_cache16_cudnnbench_b192` | 5000 | 192 | n/a | n/a | n/a | 2.08 | 10.84 GB | 86.71% | Completed C warmup baseline; 1v3 avg_pt -1.5525 |
| `dual_C_w3000_s3000_freezeonly_value_w002_clip010_cache16_cudnnbench_b192` | 3000 | 192 | n/a | n/a | n/a | 2.08 | 9.86 GB | 85.26% | C freeze-only baseline; 1v3 avg_pt -0.81 |
| `dual_C_w3000_s4000_value_w002_clip010_cache16_cudnnbench_b192` | 4000 | 192 | n/a | n/a | n/a | 2.12 | 10.57 GB | 88.36% | C warmup+1k PPO; 1v3 avg_pt -0.9 |
| `dual_D_s3000_value_w002_clip010_cache16_cudnnbench_b192_statsfix` | 3000 | 192 | n/a | n/a | n/a | 1.88 | 10.32 GB | 88.07% | Fair D 3k; 1v3 avg_pt -0.8775 |
| `dual_D_s5000_value_w002_clip010_cache16_cudnnbench_b192_oldstart_resume3000` | 5000 | 192 | n/a | n/a | n/a | 4.31 | 10.92 GB | 86.67% | Current 5k winner; 1v3 avg_pt +0.765 |
| `dual_D_s8000_value_w002_clip010_cache16_cudnnbench_b192_oldstart_resume5000` | 8000 | 192 | n/a | n/a | n/a | 4.78 | 10.95 GB | 88.53% | Completed 8k gate; 1v3 regressed to -2.6775 |
| `dual_D_s12000_value_w002_clip010_cache16_cudnnbench_b192_oldstart_resume5000` | 12000 | 192 | n/a | n/a | n/a | 3.30 | 11.34 GB | 90.40% | Completed 12k gate; 1v3 regressed to -0.99 |

Notes:

- Earlier `workers=3/4, file_batch=8` probes were run in parallel and are not
  used as single-run defaults because they inflated system memory pressure.
- `file_batch_size=10, prefetch=3` is too aggressive for the desktop's 32 GB
  RAM when dual-tower training is active.
- `batch_size=768` has enough memory headroom but lower throughput, so the
  better operating point is around `batch_size=640`.
- `dual_b640_w3_f6_p2_s3000` finished cleanly with final `val_loss=4.471712`
  and `val_corr=0.5656`. Intermediate one-batch validation points bounced, but
  the 3000-step point recovered to the best value in the window, so this is not
  treated as a training bug.
- `dual_b640_w3_f6_p2_full_s3000` used the real file pool (`train=2,462,176`,
  `val=50,248`) and finished with final `val_loss=3.266216`,
  `val_corr=0.4642`. A previous 30k attempt with `max_train_files=512` was
  stopped because it was a resource-probe-sized training pool, not a valid
  mainline pretrain.
- `dual_b640_w2_f6_p2_full_s30000` used the real file pool and completed cleanly
  in `8004.6s`. Best validation was at step `19500` with
  `val_loss=2.666081`, `val_corr=0.5998`; final step `30000` was
  `val_loss=2.780849`, `val_corr=0.5859`. Prefer the best checkpoint over
  `latest.pth` for D/E online initialization.
- Process-tree CPU/GPU-memory attribution is best-effort on Windows; use global
  GPU memory, global GPU utilization, process-tree RSS/private memory, and
  system RAM as the authoritative resource-safety signals.
- Online runner resource guard now treats the maximum system RAM observed across
  role snapshots as the sample-level system RAM, so a trainer/client high-water
  reading cannot be hidden by the first role sampled.
- Online client now retries transient server `connect()` failures with a bounded
  wait and restores blocking mode after connect; this prevents a single TCP
  timeout from aborting an otherwise healthy long run.
- `replay_is/version_gap_max` now records the actual 500-step window maximum in
  new runs. Older runs before this fix under-report that tag as a batch-max
  average, so use it only as a rough staleness signal there.
- `dual_E_w3000_s12000_value_w002_cache16_cudnnbench_b192_resume5000` stayed
  below the hard guard and finished with errors false, but system RAM still
  touched `90.42%`; keep batch `192` as the long-window default.
- `dual_E_w3000_s8000_value_w002_cache16_cudnnbench_b192_resume5000` stayed
  just below the hard RAM guard with max system RAM `90.76%`; it is resource-safe
  but leaves little RAM headroom.
- `dual_E_w0_s9000_value_w002_cache16_cudnnbench_b192_resume8000` was lighter
  because it only ran the 8k -> 9k unfreeze window; resource metrics are useful
  for short stability probes, not as a long-window memory guarantee.
- `dual_E_w0_s9000_ef047_er1e3_value_w002_cache16_cudnnbench_b192_resume8000`
  stayed resource-safe, but `entropy_floor=0.47` with `entropy_adjust_rate=0.001`
  barely changed the 8k -> 9k entropy/ratio trajectory and did not improve
  same-seed `1v3`.
- `dual_E_w0_s9000_clip010_value_w002_cache16_cudnnbench_b192_resume8000_retry`
  stayed resource-safe and is the first 8k -> 9k actor-stability probe to
  improve same-seed `1v3`; continue with batch `192` and `clip_ratio=0.1`.
- Online old-policy refresh must reuse the existing old model objects and call
  `load_state_dict`; using `deepcopy` at `old_update_every` creates extra GPU
  model copies and can trip the guard.
- In C/D/E runs with reproducible seeds, `allow_cudnn_benchmark=true` is needed
  for the desktop resource profile. It trades strict deterministic convolution
  algorithm selection for substantially lower GPU memory pressure.
- For the 2026-05-25 hyperparameter search, the user allowed normal desktop
  browser/app usage during training. Those runs used a softer resource policy
  (`max_system_mem_percent=95`, `hard_system_mem_percent=98`,
  `resource_breach_samples=5`). Treat transient desktop RAM/GPU spikes below
  the hard guard as environment noise, not a configuration failure, unless the
  system stalls, crashes, or training logs show real errors.

## Evidence Paths

Probe outputs are under:

```text
logs/oracle_critic_resource_probe/
```

The most relevant summaries:

```text
logs/oracle_critic_resource_probe/dual_b640_w3_f6_p2_s500/summary.json
logs/oracle_critic_resource_probe/dual_b640_w3_f6_p2_s3000/summary.json
logs/oracle_critic_resource_probe/dual_b640_w3_f6_p2_full_s3000/summary.json
logs/oracle_critic_resource_probe/dual_b640_w2_f6_p2_full_s30000/summary.json
logs/oracle_critic_resource_probe/dual_b640_w2_f6_p2_s500/summary.json
logs/oracle_critic_resource_probe/dual_b512_w2_f6_p2_s180/summary.json
```
