# 2026-03-22 P1 Auxiliary Adjustment Log

> Historical note: this document records the pre-`2026-03-28` auxiliary search redesign.
> Mentions of `P1 solo`, `pairwise`, and `solo -> pairwise` propagation are preserved as history only.
> Current default P1 mainline is `calibration -> protocol_decide -> winner_refine -> ablation`.

## Scope

This note records the reasoning and code changes behind the latest `Stage 0.5 / P1` auxiliary-search adjustments:

- replace arbitrary `rank / opp / danger` turn weighting with evidence-based ranges
- check whether player speed and danger changed materially across years in the local dataset
- use `P1 calibration` and `P1 solo` results to decide cross-head scale and the next search range
- tighten `solo -> pairwise` survivor rules so obviously bad auxiliary families do not keep propagating

## 1. Turn-Weighting Update

The first issue was that the old turn buckets and multipliers were too arbitrary.

Public references used:

- Kobalab hand-speed stats:
  - <https://blog.kobalab.net/entry/20180118/1516202840>
- Kobalab open-hand speed stats:
  - <https://blog.kobalab.net/entry/20180203/1517667551>
- RMS strategy PDF:
  - <https://tnt-rcr.com/wordpress/wp-content/uploads/2024/07/Riichi-Mahjong-Strategy-rms.pdf>

Local sample used:

- `18` stratified years
- `60` files per year
- `1080` games
- `707,930` supervised states
- sample list: [logs/turn_weight_sample_paths.txt](../../logs/turn_weight_sample_paths.txt)
- aggregated stats: [logs/turn_weight_stats_sample.json](../../logs/turn_weight_stats_sample.json)

Chosen turn buckets:

- early: `0-4`
- mid: `5-11`
- late: `12+`

Key local bucket stats:

- `rank_match_rate = 0.4719 / 0.4676 / 0.4643`
- `opp_any_tenpai_rate = 0.0342 / 0.4078 / 0.8303`
- `opp_any_near_rate = 0.2772 / 0.8721 / 0.9817`
- `danger_state_has_any_rate = 0.0100 / 0.1650 / 0.4117`
- `danger_positive_discard_rate_given_valid = 0.0014 / 0.0291 / 0.0909`

Selected defaults:

- `rank = 1.00 / 1.05 / 1.15`
- `opp = 0.20 / 1.00 / 1.60`
- `danger = 0.05 / 1.00 / 2.50`

Interpretation:

- `rank` stays relevant across the whole hand, but only increases mildly with turn
- `opp` is light early, meaningful in midgame, and important late
- `danger` is nearly useless early and becomes sharply more important late

Implementation details:

- added turn-weight parsing and bucket weighting in `mortal/train_supervised.py`
- updated defaults in `mortal/config.toml`
- updated templates in `mortal/config.example.toml`
- aligned validation slice boundaries to the same `0-4 / 5-11 / 12+` split so training weights and monitoring slices use the same turn semantics

## 2. Year-by-Year Drift Check

The next question was whether player speed and danger changed materially over time.

Local year-trend sample:

- `18` years
- `120` files per year
- around `76k-82k` states per year
- selected files: [logs/year_trend_sample_paths.txt](../../logs/year_trend_sample_paths.txt)
- summary stats: [logs/year_trend_stats_sample.json](../../logs/year_trend_stats_sample.json)

Main finding:

- there is no strong secular trend that players are getting much faster
- danger does not start materially earlier in recent years
- the most visible drift is only a small increase in late dangerous-discard density

Early five-year average (`2010-2014`) vs recent five-year average (`2021-2025`):

- `turn 7 opp_any_tenpai_rate: 35.9% -> 34.1%`
- `turn 10 opp_any_tenpai_rate: 64.0% -> 63.3%`
- `mid 5-11 opp_any_tenpai_rate: 41.34% -> 40.35%`
- `late 12+ danger_state_has_any_given_valid_rate: 50.33% -> 49.03%`
- `late 12+ danger_positive_discard_rate_given_valid: 9.05% -> 9.23%`

Stable crossing points:

- `opp_any_tenpai_rate >= 50%` stays around turn `9`
- `danger_positive_discard_rate_given_valid >= 5%` stays around turn `10`

Conclusion:

- no evidence for year-conditioned turn buckets
- the same `0-4 / 5-11 / 12+` split remains defensible across years

## 3. Cross-Head Scale Should Not Use Raw Loss Directly

The user then asked how to compare `rank`, `opp`, and `danger` against each other.

Important point:

- raw loss magnitudes are not directly comparable across these heads
- `rank` already includes a sample-dependent weight template
- `opp` and `danger` use different target structures and sparsity

The correct `P1 calibration` logic is:

- `rank_effective = rank_aux_weight_mean * rank_aux_raw_loss`
- `opp_effective = opponent_aux_loss`
- `danger_effective = danger_aux_loss`
- also record trunk-side gradient pressure using `phi_grad_rms`
- blend loss-based and gradient-based calibration with a geometric mean

Relevant calibration output was read from:

- [logs/stage05_fidelity/s05_fidelity_main/state.json](../../logs/stage05_fidelity/s05_fidelity_main/state.json)

Calibration result:

- `rank_effective_base = 0.0685`
- `opp_effective_per_unit = 1.1051`
- `danger_effective_per_unit = 0.2655`
- `rank_grad_effective_base = 1.09e-6`
- `opp_grad_effective_per_unit = 1.64e-5`
- `danger_grad_effective_per_unit = 9.55e-6`
- `opp_weight_per_budget_unit = 0.064`
- `danger_weight_per_budget_unit = 0.144`
- `joint_combo_factor = 0.883`

Interpretation:

- under equal effective budget, `danger` needs a larger explicit head weight than `opp`
- this does not mean `danger` is intrinsically more important than `opp`
- it means `danger` produces less effective loss/gradient pressure per unit weight, so it needs a larger coefficient to reach a comparable training budget

## 4. What the Initial P1 Solo Round Showed

The initial `P1 solo` search used the old budget ranges:

- `rank: [0.25, 0.5, 1.0, 1.5]`
- `opp: [0.25, 0.5, 0.75, 1.0]`
- `danger: [0.25, 0.5, 0.75, 1.25]`

Results were read from:

- [logs/stage05_ab/s05_fidelity_main_p1_solo_s20261817](../../logs/stage05_ab/s05_fidelity_main_p1_solo_s20261817)

Across the four protocols, even the smallest solo budget already regressed `full_recent_loss`:

- `rank @ 0.25`: mean delta `+0.01693`
- `opp @ 0.25`: mean delta `+0.01971`
- `danger @ 0.25`: mean delta `+0.00898`

This means the old lowest budget was already too heavy for all three families.

Relative damage ordering at the old minimum:

- least harmful: `danger`
- next: `rank`
- most harmful: `opp`

Important nuance:

- `danger` still looked most promising among the three because its loss damage was the smallest and its scenario-side gains were the strongest
- this supports keeping `danger` in the search, but only at much smaller weights

## 5. Code Changes Made After Reviewing Solo Results

### 5.1 First correction: shrink the solo search

The first correction was to move all solo ranges below the old minimum, because the old minimum was already too heavy.

### 5.2 Second correction: make the ranges asymmetric

After reviewing the old solo results and the calibration output together, the search was revised again so that:

- `rank` and `opp` search narrower and lighter
- `danger` keeps a slightly wider and higher ceiling

Current solo budget ranges in `mortal/run_stage05_fidelity.py`:

- `rank: [0.03, 0.06, 0.10, 0.15]`
- `opp: [0.03, 0.06, 0.10, 0.15]`
- `danger: [0.05, 0.10, 0.20, 0.30]`

Using the current calibration mapping, these are roughly:

- `opp_weight = 0.064 * budget_ratio`
- `danger_weight = 0.144 * budget_ratio`

So the actual searched head-weight bands are approximately:

- `opp: 0.0019 / 0.0038 / 0.0064 / 0.0096`
- `danger: 0.0072 / 0.0144 / 0.0288 / 0.0432`

The point is not raw coefficient symmetry; the point is closer effective-budget fairness under the observed calibration.

### 5.3 Third correction: stop bad solo families from leaking into pairwise

Before this change, `select_p1_family_survivors()` would keep the best candidate in each family even if that family was still clearly worse than the same protocol's `ce_only` baseline.

This was later finalized under the canonical `P1 policy_quality` rule so that:

- a family only survives from `solo` into `pairwise`
- if it is valid
- and it stays inside the canonical `comparison_recent_loss = recent_policy_loss` gate
- and, when available, it also stays inside the `old_regression_policy_loss` guardrail with epsilon `0.0035`
- equivalently, a family winner must not lose clearly to the same protocol's `ce_only` under that same `policy_quality` gate

This matters because otherwise the pipeline wastes pairwise search budget on families that already failed the basic gate.

## 6. Files Touched

Main training and weighting changes:

- `mortal/train_supervised.py`
- `mortal/config.toml`
- `mortal/config.example.toml`
- `mortal/test_train_supervised.py`

P1 search-space and survivor-rule changes:

- `mortal/run_stage05_fidelity.py`
- `mortal/test_stage05_fidelity.py`

## 7. Verification

Checks run after the code changes:

- `python -m unittest mortal.test_train_supervised`
- `python -m py_compile mortal/train_supervised.py mortal/test_train_supervised.py`
- `python -m unittest mortal.test_stage05_fidelity`
- `python -m py_compile mortal/run_stage05_fidelity.py mortal/test_stage05_fidelity.py`

All of the above passed in the local `mortal` environment.

## 8. Current Practical Recommendation

At the current evidence level:

- keep `ce_only` as the main baseline
- rerun `P1 solo` with the new asymmetric micro-budget ranges
- only allow families that satisfy the canonical `policy_quality` gate and do not lose clearly to protocol-local `ce_only` to continue

If the rerun still shows no surviving family:

- keep `Stage 0.5` mainline as `ce_only`
- treat auxiliary heads as later-stage or highly constrained follow-up experiments
- among the three, `danger` remains the most reasonable family to test first under very light settings
