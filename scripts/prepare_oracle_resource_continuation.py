"""Freeze a resource-only continuation of an existing Oracle scientific arm."""
import argparse
from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
import shutil

from profile_oracle_runtime import digest, dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rayon', type=int, default=4)
    parser.add_argument('--prefetch', type=int, default=2)
    parser.add_argument('--windows-high-qos', action='store_true')
    args = parser.parse_args()
    import toml
    parent, output = Path(args.parent).resolve(), Path(args.output).resolve()
    if output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError('resource continuation must have a separate run root')
    original = json.loads((parent / 'runtime_manifest.json').read_text(encoding='utf-8'))
    source = Path(original['source_root'])
    for name, expected in original['source_sha256'].items():
        if digest(source / name) != expected:
            raise ValueError('parent source changed: ' + name)
    if digest(original['config']['path']) != original['config']['sha256']:
        raise ValueError('parent configuration changed')
    output.mkdir(parents=True, exist_ok=False)
    frozen = output / 'source'
    for name in original['source_sha256']:
        target = frozen / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    relative = 'mortal/online/pretrain_oracle_critic.py'
    before = (source / relative).read_text(encoding='utf-8')
    after = Path(relative).read_text(encoding='utf-8')
    # Keep the parent's native-module provenance behavior. The main worktree has
    # an independent package-resolution change outside this resource experiment.
    after = after.replace('sha256_file, native_module_file', 'sha256_file')
    after = after.replace('native_file=native_module_file(libriichi)', 'native_file=libriichi.__file__')
    (frozen / relative).write_text(after, encoding='utf-8', newline='\n')
    resource_helper = 'mortal/core/process_resources.py'
    shutil.copyfile(resource_helper, frozen / resource_helper)
    (output / 'resource_source.diff').write_text('\n'.join(difflib.unified_diff(
        before.splitlines(), after.splitlines(), fromfile='parent/' + relative,
        tofile='continuation/' + relative, lineterm='')) + '\n', encoding='utf-8')
    runner = frozen / 'scripts/run_sl_rl_repair_pipeline.py'
    code = runner.read_text(encoding='utf-8')
    anchor = "    env['PYTHONPATH'] = str(source)\n"
    if code.count(anchor) != 1:
        raise ValueError('unexpected parent runner')
    if "manifest.get('runtime_environment', {})" not in code:
        code = code.replace(anchor, anchor + "    for key, value in manifest.get('runtime_environment', {}).items():\n"
                            "        if key not in ('RAYON_NUM_THREADS',):\n"
                            "            raise ValueError('unsupported runtime resource environment: ' + key)\n"
                            "        env[key] = str(value)\n")
    runner.write_text(code, encoding='utf-8', newline='\n')
    cfg = toml.loads(Path(original['config']['path']).read_text(encoding='utf-8'))
    pre = cfg['oracle_critic_pretrain']
    changes = {}
    for key, value in tuple(pre.items()):
        if isinstance(value, str) and value.startswith(str(parent)):
            replacement = str(output) + value[len(str(parent)):]
            changes[key] = {'before': value, 'after': replacement}
            pre[key] = replacement
    for key, value in {'rayon_num_threads': args.rayon, 'prefetch_factor': args.prefetch,
                       'train_torch_num_threads': 1, 'eval_torch_num_threads': 1,
                       'cuda_memory_fraction': .72,
                       'windows_high_qos': args.windows_high_qos,
                       'eval_prefilter_games': False}.items():
        changes[key] = {'before': pre.get(key), 'after': value}
        pre[key] = value
    checkpoint_root = output / 'critic/checkpoints'
    checkpoint_root.mkdir(parents=True)
    for item in (parent / 'critic/checkpoints').iterdir():
        if item.is_file():
            shutil.copyfile(item, checkpoint_root / item.name)
    shutil.copyfile(parent / 'critic/metrics.jsonl', output / 'critic/metrics.jsonl')
    initial = output / 'frozen_initial_sl_policy.pth'
    shutil.copyfile(original['initial_checkpoint']['path'], initial)
    for name in ('matched_protocol.json', 'warm_control_final_audit.json'):
        if (parent / name).exists():
            shutil.copyfile(parent / name, output / name)
    config_file = output / 'critic_config.toml'
    config_file.write_text(toml.dumps(cfg), encoding='utf-8', newline='\n')
    manifest = dict(original)
    manifest.update(source_root=str(frozen),
                    source_sha256={name: digest(frozen / name) for name in
                                   sorted({*original['source_sha256'], resource_helper})},
                    config={'path': str(config_file), 'sha256': digest(config_file)},
                    initial_checkpoint={'path': str(initial), 'sha256': digest(initial)},
                    latest_checkpoint=str(checkpoint_root / 'latest.pth'),
                    runtime_environment={'RAYON_NUM_THREADS': str(args.rayon)},
                    resource_continuation={'parent_run': str(parent),
                        'parent_manifest_sha256': digest(parent / 'runtime_manifest.json'),
                        'resume_checkpoint_sha256': digest(parent / 'critic/checkpoints/latest.pth'),
                        'resource_changes': changes, 'scientific_arm_unchanged': True,
                        'benchmark_checkpoints_adopted': False,
                        'created_at': datetime.now(timezone.utc).isoformat()})
    dump(output / 'runtime_manifest.json', manifest)
    spec = json.loads((parent / 'apex_supervisor_spec.json').read_text(encoding='utf-8'))
    for key, value in spec.items():
        if isinstance(value, str):
            spec[key] = value.replace(str(parent), str(output))
        elif isinstance(value, list):
            spec[key] = [part.replace(str(parent), str(output)) if isinstance(part, str) else part for part in value]
    dump(output / 'apex_supervisor_spec.json', spec)
    dump(output / 'continuation_prepared.json', {'state': 'prepared_not_launched',
                                               **manifest['resource_continuation']})
    print(json.dumps({'output': str(output), 'state': 'prepared_not_launched', 'resource_changes': changes}))


if __name__ == '__main__':
    main()
