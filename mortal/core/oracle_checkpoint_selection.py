"""Checkpoint roles for opt-in Oracle primary-with-diagnostics selection.

Observation is not promotion: a non-gate validation may save a candidate but
cannot overwrite the paired-evidence accepted checkpoints.
"""
from __future__ import annotations

import math


def checkpoint_selection(*, primary_loss, all_players_loss, best_primary_loss,
                         best_val_loss, best_observed_primary_loss,
                         primary_with_diagnostics, adaptive_active,
                         gate_observed, accepted):
    if not math.isfinite(primary_loss) or not math.isfinite(all_players_loss):
        raise ValueError('checkpoint selection metrics must be finite')
    eligible = (accepted if adaptive_active and
                (primary_with_diagnostics or gate_observed) else True)
    return {
        'best': eligible and all_players_loss < best_val_loss,
        'best_primary': eligible and primary_loss < best_primary_loss,
        'best_observed_primary': primary_with_diagnostics
        and primary_loss < best_observed_primary_loss,
    }


def baseline_checkpoint_roles(primary_with_diagnostics):
    return (('latest', 'adaptive_best', 'best', 'best_primary', 'best_observed_primary')
            if primary_with_diagnostics else ('latest', 'adaptive_best'))
