import json
import sys
from pathlib import Path
sys.path.insert(0, r"C:\Users\numbe\Desktop\MahjongAI\mortal")
import run_sl_ab as ab
import run_sl_loader_ab as loader_ab
import run_sl_fidelity as fidelity
import sl_current_defaults as sl_defaults

# Historical explicit p-core benchmark helper. This is not a default runtime path.

suite_name = 'sl_loader_ab_nw6_pcores_20260326_2208'
summary_path = Path(r'C:\Users\numbe\Desktop\MahjongAI\logs\sl_loader_ab') / suite_name / 'summary.json'
grouped = ab.group_files_by_month(ab.load_all_files())
eval_splits = ab.build_eval_splits(grouped, 20260326 + 77, {'full_recent': 64, 'old_regression': 32})
candidate = loader_ab.load_reference_candidate(
    run_name='sl_fidelity_p1_top3_20260325_022900',
    protocol_arm=sl_defaults.CURRENT_PRIMARY_PROTOCOL_ARM,
    rank_budget_ratio=0.15,
    opp_budget_ratio=0.03,
    danger_budget_ratio=0.10,
)
summary_payload = {
    'suite_name': suite_name,
    'reference_candidate': {
        'arm_name': candidate.arm_name,
        'protocol_arm': candidate.meta.get('protocol_arm'),
        'rank_budget_ratio': candidate.meta.get('rank_budget_ratio'),
        'opp_budget_ratio': candidate.meta.get('opp_budget_ratio'),
        'danger_budget_ratio': candidate.meta.get('danger_budget_ratio'),
    },
    'benchmark_settings': {
        'seed': 20260326,
        'step_scale': 0.25,
        'benchmark_phase': 'phase_a',
        'val_every_steps': 750,
        'monitor_val_batches': 64,
        'full_recent_files': 64,
        'old_regression_files': 32,
        'affinity': 'p_cores',
        'scope': 'nw6_only',
    },
    'rounds': {},
    'best_so_far': None,
}
fidelity.atomic_write_json(summary_path, summary_payload)
configs = [
    loader_ab.make_loader_config(num_workers=6, file_batch_size=5, prefetch_factor=3, val_file_batch_size=7, val_prefetch_factor=2),
    loader_ab.make_loader_config(num_workers=6, file_batch_size=6, prefetch_factor=3, val_file_batch_size=7, val_prefetch_factor=2),
    loader_ab.make_loader_config(num_workers=6, file_batch_size=5, prefetch_factor=2, val_file_batch_size=7, val_prefetch_factor=2),
    loader_ab.make_loader_config(num_workers=6, file_batch_size=6, prefetch_factor=2, val_file_batch_size=7, val_prefetch_factor=2),
    loader_ab.make_loader_config(num_workers=6, file_batch_size=7, prefetch_factor=2, val_file_batch_size=7, val_prefetch_factor=2),
    loader_ab.make_loader_config(num_workers=6, file_batch_size=8, prefetch_factor=2, val_file_batch_size=7, val_prefetch_factor=2),
]
round_payload = loader_ab.run_round(
    suite_name=suite_name,
    round_name='train_scan_nw6_pcores',
    configs=configs,
    grouped=grouped,
    eval_splits=eval_splits,
    candidate=candidate,
    seed=20260326,
    step_scale=0.25,
    val_every_steps=750,
    monitor_val_batches=64,
    full_recent_files=64,
    old_regression_files=32,
    summary_payload=summary_payload,
    summary_path=summary_path,
    phase_name='phase_a',
)
summary_payload['overall_best'] = loader_ab.choose_best_stable(round_payload['results'])
fidelity.atomic_write_json(summary_path, summary_payload)
print(json.dumps(fidelity.normalize_payload(summary_payload), ensure_ascii=False, indent=2))
