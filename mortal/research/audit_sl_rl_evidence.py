"""Read-only evidence collection for the September 2026 SL/RL audit.

This never runs inference, changes a checkpoint, or opens sealed test data.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from mortal.core.artifacts import atomic_write_json, file_sha256


def tensor_state_sha256(state):
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode('utf-8'))
        digest.update(str((value.dtype, tuple(value.shape))).encode('ascii'))
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def checkpoint_summary(path):
    import torch

    before = path.stat()
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    cfg = state.get('config', {})
    result = {
        'path': str(path.resolve()),
        'size': before.st_size,
        'mtime_ns': before.st_mtime_ns,
        'steps': state.get('steps'),
        'optimizer_steps': state.get('optimizer_steps'),
        'checkpoint_id': state.get('checkpoint_id'),
        'config': {k: cfg[k] for k in ('env', 'policy', 'value', 'supervised', 'resnet', 'oracle_critic_pretrain') if k in cfg},
        'oracle_critic_pretrain': state.get('oracle_critic_pretrain'),
        'checkpoint_provenance': state.get('checkpoint_provenance'),
        'component_hashes': {
            k: tensor_state_sha256(state[k])
            for k in ('mortal', 'policy_net', 'oracle_brain', 'value_net')
            if isinstance(state.get(k), dict)
        },
        'metrics': {k: v for k, v in state.items() if 'metrics' in k and isinstance(v, dict)},
        'optimizer_lrs': [g.get('lr') for g in state.get('optimizer', {}).get('param_groups', [])],
    }
    adaptive = state.get('adaptive_curriculum_state', state.get('adaptive_curriculum', {}))
    if isinstance(adaptive, dict):
        result['adaptive'] = {k: v for k, v in adaptive.items() if 'cluster_records' not in k}
    after = path.stat()
    result['file_stable_during_read'] = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = Path(args.output_dir).resolve()
    if (output / 'evidence_before.json').exists():
        raise FileExistsError('choose a new output directory to preserve the existing evidence snapshot')
    output.mkdir(parents=True, exist_ok=True)

    import torch
    import toml
    torch.set_num_threads(1)
    evidence = {'captured_at': datetime.now(timezone.utc).isoformat(), 'root': str(root)}
    for name, command in [('git_head', ['git', 'rev-parse', 'HEAD']), ('git_status', ['git', 'status', '--porcelain'])]:
        evidence[name] = subprocess.check_output(command, cwd=root, text=True, encoding='utf-8')
    source_files = [
        'mortal/online/train_online.py', 'mortal/online/pretrain_oracle_critic.py',
        'mortal/supervised/train_supervised.py', 'mortal/supervised/run_sl_ab.py',
        'mortal/core/adaptive_curriculum.py', 'mortal/supervised/adaptive_curriculum.py',
        'mortal/data/dataloader.py', 'mortal/data/oracle_value.py',
        'mortal/eval/one_vs_three.py', 'mortal/eval/engine.py',
        'libriichi/src/dataset/invisible.rs', 'libriichi/src/dataset/gameplay.rs',
    ]
    evidence['source_sha256'] = {p: file_sha256(root / p) for p in source_files}
    for p in source_files:
        dest = output / 'source_before' / p
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes((root / p).read_bytes())
    evidence['torch_version'] = torch.__version__
    from libriichi import libriichi as native
    evidence['libriichi'] = {'path': native.__file__, 'sha256': file_sha256(native.__file__)}

    cde = root / 'logs/oracle_cde'
    checkpoint_relpaths = {
        'sl_canonical': 'mortal/checkpoints/sl_canonical.pth',
        's70': 'logs/sl_fidelity/sl_anchor_longabc_s70_20260609_r1_1v3_compare/best_action_score.pth',
        'e9k': 'logs/oracle_cde/dual_E_w0_s9000_clip010_value_w002_cache16_cudnnbench_b192_resume8000_retry/checkpoints/mortal.pth',
        'e10k_policy_stop': 'logs/oracle_cde/dual_E_w0_s10000_clip010_policystop_value_w002_cache16_cudnnbench_b192_resume9000/checkpoints/mortal.pth',
        'd5k': 'logs/oracle_cde/dual_D_s5000_value_w002_clip010_cache16_cudnnbench_b192_oldstart_resume3000/checkpoints/mortal.pth',
    }
    evidence['checkpoints'] = {}
    for label, relative in checkpoint_relpaths.items():
        path = root / relative
        if path.exists():
            evidence['checkpoints'][label] = checkpoint_summary(path)
        else:
            evidence['checkpoints'][label] = {'path': str(path), 'missing': True}
        print(f'checkpoint audited: {label}', flush=True)

    evidence['cde_evaluation_inventory'] = []
    for directory in sorted(cde.glob('*1v3*')):
        if not directory.is_dir():
            continue
        cfg_path = directory / 'config.toml'
        cfg = toml.load(cfg_path) if cfg_path.exists() else {}
        results = [json.loads(p.read_text('utf-8')) for p in directory.glob('worker_result*.json')]
        evidence['cde_evaluation_inventory'].append({
            'name': directory.name,
            'eval_config': cfg.get('1v3'),
            'raw_log_count': sum(1 for _ in directory.rglob('*.json.gz')),
            'worker_results': results,
        })

    playoff = root / 'logs/sl_fidelity/sl_formal_triplet_20260405_winner_playoff_1v3/formal_1v3_round.json'
    evidence['sl_formal_playoff'] = json.loads(playoff.read_text('utf-8'))
    sl_tree = ast.parse((root / 'mortal/supervised/run_sl_ab.py').read_text('utf-8-sig'))
    evidence['sl_data_windows'] = next(ast.literal_eval(n.value) for n in sl_tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'WINDOWS' for t in n.targets))

    oracle_root = root / 'logs/oracle_critic_formal/s70_broad_to_recent_strong24m12m_adaptive_sf200_wd0_20260901_r1'
    oracle_case = oracle_root / 'phases/phase_a/sf_lr200_wd000'
    cfg = toml.load(oracle_case / 'config.toml')
    evidence['active_oracle_config'] = {k: cfg[k] for k in ('env', 'oracle_critic_pretrain', 'policy', 'value')}
    evidence['oracle_curriculum_design'] = json.loads((oracle_root / 'curriculum_design.json').read_text('utf-8'))
    evidence['oracle_cache_manifest'] = json.loads((root / 'logs/oracle_event_cache/s70_broad_to_recent_strong24m12m_20260901_r1/manifest.json').read_text('utf-8'))
    evidence['active_oracle_gates'] = []
    last_metrics = None
    with (oracle_case / 'metrics.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # A concurrently appended last line may be incomplete.
            last_metrics = record
            adaptive = record.get('adaptive_curriculum', {})
            if adaptive:
                evidence['active_oracle_gates'].append({
                    'steps': record.get('steps'),
                    'adaptive': {k: v for k, v in adaptive.items() if k not in ('state', 'cluster_records', 'best_cluster_records')},
                    'val': {k: v for k, v in record.get('val', {}).items() if isinstance(v, (int, float, str, bool))},
                })
    if last_metrics:
        evidence['active_oracle_last_steps'] = last_metrics.get('steps')
    evidence['local_sl_monitor'] = [json.loads(p.read_text('utf-8')) for p in sorted((root / 'logs/sl_monitor').glob('*.json'))]
    destination = output / 'evidence_before.json'
    atomic_write_json(destination, evidence)
    print(f'evidence: {destination}', flush=True)


if __name__ == '__main__':
    main()
