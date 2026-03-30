from __future__ import annotations

# Post-leakage Stage 0.5 mainline frozen for downstream Stage 1 seeding.
# Keep the verified P0 top3 in historical factual order; do not reorder it to
# match later downstream protocol winners.
CURRENT_P0_TOP3_PROTOCOL_ARMS = (
    'C_A2y_cosine_broad_to_recent_strong_12m_6m',
    'C_A2x_cosine_broad_to_recent_strong_24m_12m',
    'C_A1x_cosine_broad_to_recent_mild_24m_12m',
)

# Current verified protocol_decide winner: A2x.
CURRENT_PROTOCOL_DECIDE_WINNER_ARM = 'C_A2x_cosine_broad_to_recent_strong_24m_12m'
CURRENT_PRIMARY_PROTOCOL_ARM = CURRENT_PROTOCOL_DECIDE_WINNER_ARM
CURRENT_STAGE1_TOP_PROTOCOL_ARMS = CURRENT_P0_TOP3_PROTOCOL_ARMS

# Current validated Stage 0.5 validation loader defaults on this machine.
DEFAULT_VAL_FILE_BATCH_SIZE = 8
DEFAULT_VAL_PREFETCH_FACTOR = 5

# Current slim P1 calibration policy:
# - keep a single representative protocol arm (A2y)
# - reuse the frozen 2026-03-25 post-shape single-head calibration numbers
#   instead of rerunning pure single-head probes in the current run
# - only rerun pairwise/triple probes to refresh combo factors for the current
#   P1 line
# Important: this representative arm is not the downstream protocol winner.
CURRENT_P1_CALIBRATION_REPRESENTATIVE_PROTOCOL_ARM = 'C_A2y_cosine_broad_to_recent_strong_12m_6m'
CURRENT_P1_CALIBRATION_PROTOCOL_ARMS = (
    CURRENT_P1_CALIBRATION_REPRESENTATIVE_PROTOCOL_ARM,
)
CURRENT_P1_CALIBRATION_MODE = 'combo_only'
CURRENT_P1_SINGLE_HEAD_CALIBRATION_SOURCE = (
    '2026-03-25 post-shape top3 calibration on the frozen A2y/A2x/A1x pool'
)
# Frozen single-head baseline reused verbatim by combo_only runs. These values
# are inherited into the current slim calibration path; they are not
# recomputed inside the run unless the caller switches modes.
CURRENT_P1_SINGLE_HEAD_CALIBRATION_BASELINE = {
    'mapping_mode': 'hybrid_loss_grad_geomean',
    'rank_effective_base': 0.05121494575615136,
    'opp_effective_per_unit': 0.8990692208828822,
    'danger_effective_per_unit': 0.04924819144263803,
    'rank_grad_effective_base': 8.414337671069916e-07,
    'opp_grad_effective_per_unit': 1.801891199211999e-05,
    'danger_grad_effective_per_unit': 1.5777937623084881e-06,
    'opp_weight_per_budget_unit_loss': 0.057,
    'danger_weight_per_budget_unit_loss': 0.18,
    'opp_weight_per_budget_unit_grad': 0.047,
    'danger_weight_per_budget_unit_grad': 0.18,
    'opp_weight_per_budget_unit': 0.052,
    'danger_weight_per_budget_unit': 0.18,
    'probe_weight': 0.06,
    'grad_probe_batches': 8,
    'fallback_used': False,
}

# Current protocol_decide seed2 expansion defaults. Keep the current mainline
# on the explicit flip-or-gap rule; only preserve the old ambiguity mode for
# historical run compatibility when reading archived state.
CURRENT_P1_PROTOCOL_DECIDE_PROGRESSIVE_AMBIGUITY_MODE = 'flip_or_gap'
CURRENT_P1_PROTOCOL_DECIDE_PROGRESSIVE_GAP_THRESHOLD = 0.001
CURRENT_P1_PROTOCOL_DECIDE_PROGRESSIVE_NOISE_MARGIN_MULT = 2.0

# Current winner_refine defaults frozen after the verified A2x protocol_decide
# closeout on 2026-03-30. These are explicit center arms, not an auto top-k
# rule, so future changes must update both code and docs intentionally.
CURRENT_P1_WINNER_REFINE_PROTOCOL_ARM = CURRENT_PROTOCOL_DECIDE_WINNER_ARM
CURRENT_P1_WINNER_REFINE_CENTER_MODE = 'explicit_arm_names'
CURRENT_P1_WINNER_REFINE_CENTER_ARMS = (
    'C_A2x_cosine_broad_to_recent_strong_24m_12m__B_r0046_o0037_d0037',
    'C_A2x_cosine_broad_to_recent_strong_24m_12m__B_r0034_o0014_d0041',
    'C_A2x_cosine_broad_to_recent_strong_24m_12m__B_r0052_o0025_d0043',
)
