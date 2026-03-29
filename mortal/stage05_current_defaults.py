from __future__ import annotations

# Post-leakage Stage 0.5 mainline frozen for downstream Stage 1 seeding.
CURRENT_PRIMARY_PROTOCOL_ARM = 'C_A2y_cosine_broad_to_recent_strong_12m_6m'
CURRENT_STAGE1_TOP_PROTOCOL_ARMS = (
    CURRENT_PRIMARY_PROTOCOL_ARM,
    'C_A2x_cosine_broad_to_recent_strong_24m_12m',
    'C_A1x_cosine_broad_to_recent_mild_24m_12m',
)

# Current validated Stage 0.5 validation loader defaults on this machine.
DEFAULT_VAL_FILE_BATCH_SIZE = 8
DEFAULT_VAL_PREFETCH_FACTOR = 5

# Current slim P1 calibration policy:
# - keep a single representative protocol arm (A2y)
# - reuse the frozen 2026-03-25 post-shape single-head calibration numbers
#   instead of rerunning pure single-head probes in the current run
# - only rerun pairwise/triple probes to refresh combo factors for the current
#   P1 line
CURRENT_P1_CALIBRATION_PROTOCOL_ARMS = (
    CURRENT_PRIMARY_PROTOCOL_ARM,
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
