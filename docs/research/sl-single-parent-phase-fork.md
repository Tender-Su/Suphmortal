# Single-parent B to C fork

`scripts/fork_sl_phase.py` prepares a separate C experiment from a completed,
fully saved B endpoint, including B endpoints reached through multiple same-phase
continuations. It keeps the old balanced early/late runner and same-phase entry
unchanged. It requires the parent training/evaluation source bytes unchanged.

Preparation verifies the parent completion receipt, full saved-state contract,
original A ancestor, seed manifest, fixed data index and consumed-content ledger.
It appends the direct B parent to the complete A/B chain. Learned heads, Adam
moments/internal steps, parameter mapping, scaler and cumulative auxiliary clock
are preserved. Phase counters, scheduler clock, sampler and RNG start a new C
phase. The LR is the parent's effective LR with no repeated warmup. Subsequent
C resumes restore C's own consumed cursor, RNG, scheduler and observations.

The preparation CLI requires explicit source run/checkpoint, seed manifest,
source commit, updates, observations and LR. Seed is read from the original
manifest's C entry, not inferred from a default. Use `--baseline` only for the
parent's full fixed-panel observation; its identity, successful-update count,
learned-state digest, split identity and archived checkpoint hash must match.
Reused U0 records retain their B observation source and record zero new evaluation
time. Partial observation archives are retained and require inspection rather
than being silently overwritten on restart.

The candidate-generation experiment uses one early B50k parent, C sampling
98% latest games/2% replay games, LR 1e-5, warmup 0, microbatch 512x2, inherited
auxiliary objectives and full recent512/old256 validation. Observe U0, U1k, U5k,
U10k, U20k and U50k. Intermediate curves do not stop training for nonsignificance.
The 50k successful-update endpoint is not a convergence claim or a causal C-vs-B
comparison; it does not promote a model, start 1v3, add a B control, or extend
training automatically. Existing process/resource/game guards remain external
to this entry.

Focused CPU checks: `python -m unittest mortal.tests.test_sl_phase_fork`.
