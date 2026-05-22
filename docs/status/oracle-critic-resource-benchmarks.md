# Oracle Critic Resource Benchmarks

Date: 2026-05-23

Scope: local desktop Oracle critic pretraining probes for the dual-tower `all`
train scope. Each probe used `mortal.online.pretrain_oracle_critic` through
`scripts/probe_oracle_critic_resources.py`, with short windows intended to find
a safe high-throughput operating point before longer experiments.

## Current Recommendation

Use this as the initial dual-tower pretrain default on the desktop:

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
the best balance observed between throughput and memory headroom.

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

## Probe Summary

| Run | Steps | Batch | Workers | File Batch | Prefetch | Steps/sec | GPU Mem Max | Sys Mem Max | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `dual_b512_w4_f10_p3_s120_r2` | 120 | 512 | 4 | 10 | 3 | n/a | 9.9 GB | 30.5 GB | Too close to RAM limit |
| `dual_b384_w2_f6_p2_s180` | 180 | 384 | 2 | 6 | 2 | n/a | 7.6 GB | 18.9 GB | Safe but underuses GPU |
| `dual_b512_w2_f6_p2_s180` | 180 | 512 | 2 | 6 | 2 | n/a | 9.4 GB | 19.7 GB | Safe fallback |
| `dual_b640_w2_f6_p2_s500` | 500 | 640 | 2 | 6 | 2 | 3.01 | 10.9 GB | 19.3 GB | Safe |
| `dual_b640_w3_f6_p2_s500` | 500 | 640 | 3 | 6 | 2 | 3.10 | 11.3 GB | 22.1 GB | Recommended |
| `dual_b640_w4_f6_p2_s300` | 300 | 640 | 4 | 6 | 2 | 2.82 | 10.8 GB | 24.1 GB | More RAM, slower |
| `dual_b768_w2_f6_p2_s120` | 120 | 768 | 2 | 6 | 2 | 1.99 | 12.3 GB | 20.2 GB | Slower; not worth it |

Notes:

- Earlier `workers=3/4, file_batch=8` probes were run in parallel and are not
  used as single-run defaults because they inflated system memory pressure.
- `file_batch_size=10, prefetch=3` is too aggressive for the desktop's 32 GB
  RAM when dual-tower training is active.
- `batch_size=768` has enough memory headroom but lower throughput, so the
  better operating point is around `batch_size=640`.

## Evidence Paths

Probe outputs are under:

```text
logs/oracle_critic_resource_probe/
```

The most relevant summaries:

```text
logs/oracle_critic_resource_probe/dual_b640_w3_f6_p2_s500/summary.json
logs/oracle_critic_resource_probe/dual_b640_w2_f6_p2_s500/summary.json
logs/oracle_critic_resource_probe/dual_b512_w2_f6_p2_s180/summary.json
```
