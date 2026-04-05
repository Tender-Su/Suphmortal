# 2026-03-25 A2y Auxiliary Shape Freeze

> Historical context note: this document freezes the internal shapes only.
> References to later `P1 solo / pairwise / joint refine` are historical wording from before the `2026-03-28` redesign.
> The current mainline keeps these frozen shapes but now uses:
> `calibration -> protocol_decide -> winner_refine`.
> `P1 ablation` is now a manual backlog confirmation path rather than a default mainline stage.

## Scope

This note records how the auxiliary-head internal shapes were frozen for the post-leakage `A2y` line.

The concrete question was not “should `rank / opp / danger` enter the mainline now”, but:

- if a family remains in later `P1 solo / pairwise / joint refine`,
- which **internal shape** should be treated as the default,
- so later search only needs to sweep the **family total weight**.

Authoritative run artifacts:

- micro-AB run dir:
  [logs/stage05_fidelity/s05_fidelity_a2y_internal_mix_micro1s_20260324_224456](/C:/Users/numbe/Desktop/MahjongAI/logs/stage05_fidelity/s05_fidelity_a2y_internal_mix_micro1s_20260324_224456)
- corrected summary:
  [final_summary_policy_corrected.json](/C:/Users/numbe/Desktop/MahjongAI/logs/stage05_fidelity/s05_fidelity_a2y_internal_mix_micro1s_20260324_224456/final_summary_policy_corrected.json)
- raw launcher log:
  [launcher.stdout.log](/C:/Users/numbe/Desktop/MahjongAI/logs/stage05_fidelity/launch_a2y_internal_mix_micro_20260324_224454/launcher.stdout.log)

Supporting evidence inputs:

- large-sample heuristic audit:
  [logs/aux_heuristic_audit_18k.md](/C:/Users/numbe/Desktop/MahjongAI/logs/aux_heuristic_audit_18k.md)
- auxiliary subhead gradient audit:
  [logs/aux_subhead_gradient_audit.md](/C:/Users/numbe/Desktop/MahjongAI/logs/aux_subhead_gradient_audit.md)

## Why This Was Run

The previous local evidence had two gaps:

- the repo still used heuristic internal defaults:
  - `rank = 1.4 / 1.8 / 4000 / 1.5`
  - `opp = 1.0 / 1.0`
  - `danger = 0.45 / 0.35 / 0.20`
- the old `P1 solo` evidence came from the historical `s05_fidelity_main` protocol pool rather than the new post-leakage `A2y` mainline

So the intended workflow became:

1. use `18k` statistics and gradient audit to shrink each family's internal shape grid to a tiny set
2. run a cheap `A2y`-only micro-AB
3. freeze the internal shape defaults
4. return to normal `P1 solo`, which should then sweep only family total weights

This means the micro-AB should be treated as a **shape pre-filter for `P1 solo`**, not as a new permanent stage.

## The Important Metric Correction

The first reading of this run was wrong because it focused on `full_recent_loss`.

That is not the correct winner metric for these `P1` auxiliary comparisons.

Code path:

- `policy_quality` uses `recent_policy_loss` as the comparison loss:
  [run_stage05_fidelity.py](/C:/Users/numbe/Desktop/MahjongAI/mortal/run_stage05_fidelity.py#L537)
- entry summaries still record `full_recent_loss`, but also record `recent_policy_loss`:
  [run_stage05_fidelity.py](/C:/Users/numbe/Desktop/MahjongAI/mortal/run_stage05_fidelity.py#L669)
- round ranking stores the real comparison field as `comparison_recent_loss`:
  [run_stage05_fidelity.py](/C:/Users/numbe/Desktop/MahjongAI/mortal/run_stage05_fidelity.py#L923)
- `P1 solo` survivor gating compares family winner vs protocol-local `CE-only` using that comparison loss plus old-regression guardrail:
  [run_stage05_fidelity.py](/C:/Users/numbe/Desktop/MahjongAI/mortal/run_stage05_fidelity.py#L2783)

Practical meaning:

- `full_recent_loss` on aux-enabled runs includes auxiliary tax and remains useful as a diagnostic
- but `P1 solo` winner judgment must use:
  - `comparison_recent_loss = recent_policy_loss`
  - then `selection_tiebreak_key = selection_quality_score -> -recent_policy_loss -> -old_regression_policy_loss`

This correction changed the interpretation of the micro-AB from “almost everything regressed” to “some families improved on the real gate, others only looked bad because total loss rose”.

## Candidate Shapes Before the Run

### Rank

Legacy/default heuristic shape:

- `south_factor = 1.4`
- `all_last_factor = 1.8`
- `gap_focus_points = 4000`
- `gap_close_bonus = 1.5`

Evidence-backed candidates from `18k` audit:

- round-only proxy:
  - `south_factor ≈ 1.59`
  - `all_last_factor ≈ 1.617`
  - `gap_focus_points = 4000`
  - `gap_close_bonus = 0.0`
- wide-gap proxy:
  - `south_factor ≈ 1.59`
  - `all_last_factor ≈ 1.617`
  - `gap_focus_points = 12000`
  - `gap_close_bonus ≈ 1.531`

### Opponent State

Legacy/default heuristic shape:

- `shanten = 1.0`
- `tenpai = 1.0`

Evidence-backed candidate shapes:

- `18k` stat mix:
  - normalized ratio `0.4566 / 0.5434`
  - scaled to preserve the old sum-`2.0` semantics in code:
    `0.9132674961 / 1.0867325039`
- hybrid loss+grad mix:
  - normalized ratio `0.4253 / 0.5747`
  - scaled to preserve the old sum-`2.0` semantics:
    `0.8506568408 / 1.1493431592`

### Danger

Legacy/default heuristic shape:

- `any / value / player = 0.45 / 0.35 / 0.20`

Evidence-backed candidate shapes:

- `18k` stat mix:
  - `0.0904217947 / 0.8180402859 / 0.0915379194`
- hybrid loss+grad mix:
  - `0.0407753423 / 0.9157590705 / 0.0434655872`

## Micro-AB Setup

The micro-AB used:

- protocol arm:
  `C_A2y_cosine_broad_to_recent_strong_12m_6m`
- one seed:
  `20260312`
- reduced budget:
  `step_scale = 0.20`
- one shared `CE-only` baseline
- three tiny rounds:
  - `opp_internal_mix_round`
  - `rank_shape_round`
  - `danger_internal_mix_round`

The point was to isolate shape, not to re-run a full multi-seed `P1 solo`.

## Results Under The Correct P1 Solo Metric

Baseline:

- `CE-only policy_loss = 0.6319450849`

### Opponent State

Relative to `CE-only`:

- `HYBRID_GRAD`: `policy +0.001124`, `action +0.000114`, `selection +0.000163`
- `EQ_CURRENT`: `policy +0.000498`, `action -0.000388`, `selection -0.000998`
- `18K_STAT`: `policy +0.002613`, `action -0.000609`, `selection -0.001044`

Interpretation:

- by raw policy loss alone, `EQ_CURRENT` regressed least
- by the actual `P1 solo` selection order after gate, `HYBRID_GRAD` wins because it stays inside the policy gate and has the best action/selection quality

Frozen default:

- `opp = HYBRID_GRAD = shanten 0.8506568408 / tenpai 1.1493431592`

### Rank

Relative to `CE-only`:

- `18K_ROUND_ONLY`: `policy -0.000839`, `action -0.000228`, `selection +0.000222`
- `18K_WIDE_GAP`: `policy +0.000202`, `action -0.000397`, `selection +0.000068`
- `CURRENT`: `policy +0.000194`, `action -0.000175`, `selection -0.000120`

Interpretation:

- `18K_ROUND_ONLY` is the only rank shape that clearly improves the real comparison loss
- the wide-gap proxy did not survive translation from proxy statistics to downstream training

Frozen default:

- `rank = 18K_ROUND_ONLY = south 1.59 / all-last 1.617 / gap 4000 / bonus 0.0`

### Danger

Relative to `CE-only`:

- `18K_STAT`: `policy +0.000443`, `action +0.000257`, `selection +0.000310`
- `HYBRID_GRAD`: `policy +0.000899`, `action -0.000351`, `selection -0.000316`
- `CURRENT`: `policy +0.000905`, `action -0.000354`, `selection -0.000834`

Interpretation:

- the `18k` statistical mix is the best danger shape by both action and selection quality
- danger still needs later total-weight tuning, but its internal mix no longer needs to stay at the old heuristic default

Frozen default:

- `danger = 18K_STAT = any 0.0904217947 / value 0.8180402859 / player 0.0915379194`

## What Is Frozen vs Still Open

Frozen after this run:

- `rank` internal shape
- `opp` internal shape
- `danger` internal shape
- the interpretation rule that `P1` auxiliary winner judgment must use `policy_quality`, not raw `full_recent_loss`

Still open after this run:

- family total weights
- whether each family remains in the final long-budget Stage 1 mainline
- pairwise / joint interactions between families
- `danger_ramp_steps` as a stability knob

## Operational Consequence

From this point onward, default `P1 solo` should not reopen these shape axes unless there is explicit new contrary evidence.

The intended next-step search should be:

- keep shape fixed
- sweep total family weight
- judge by the normal `P1 solo` gate:
  - `comparison_recent_loss`
  - then `selection_tiebreak_key = selection_quality_score -> -recent_policy_loss -> -old_regression_policy_loss`
  - with `old_regression` guardrail

## Code / Doc Freeze Applied

The following were updated to use the frozen defaults:

- [config.toml](/C:/Users/numbe/Desktop/MahjongAI/mortal/config.toml)
- [config.example.toml](/C:/Users/numbe/Desktop/MahjongAI/mortal/config.example.toml)
- [run_stage05_fidelity.py](/C:/Users/numbe/Desktop/MahjongAI/mortal/run_stage05_fidelity.py)
- [run_rank_shape_probe.py](/C:/Users/numbe/Desktop/MahjongAI/mortal/run_rank_shape_probe.py)
- [train_supervised.py](/C:/Users/numbe/Desktop/MahjongAI/mortal/train_supervised.py)
- [current-plan.md](/C:/Users/numbe/Desktop/MahjongAI/docs/agent/current-plan.md)
- [supervised-verified-status.md](/C:/Users/numbe/Desktop/MahjongAI/docs/status/supervised-verified-status.md)
