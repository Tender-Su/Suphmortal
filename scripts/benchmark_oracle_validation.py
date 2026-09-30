"""Compare complete fixed-monitor Oracle evaluation without touching sealed test."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time

from profile_oracle_runtime import Sampler, digest, dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--rayon', type=int, default=4)
    parser.add_argument('--file-batch', type=int, default=2)
    parser.add_argument('--early-filter', action='store_true')
    parser.add_argument('--files', type=int, default=0)
    parser.add_argument('--gpu-fraction', type=float, default=.60)
    parser.add_argument('--resident-optimizer', action='store_true')
    args = parser.parse_args()
    if not 0 < args.gpu_fraction <= .8:
        parser.error('GPU fraction must be in (0, 0.8]')
    source_run, output = Path(args.source_run).resolve(), Path(args.output).resolve()
    if output.is_relative_to(source_run) or source_run.is_relative_to(output):
        raise ValueError('outputs must not overlap the scientific run')
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source_run / 'runtime_manifest.json').read_text(encoding='utf-8'))
    source = Path(manifest['source_root'])
    if digest(source_run / 'critic_config.toml') != manifest['config']['sha256']:
        raise ValueError('source configuration changed')
    os.environ.update(MORTAL_CFG=str(source_run / 'critic_config.toml'),
                      RAYON_NUM_THREADS=str(args.rayon), PYTHONDONTWRITEBYTECODE='1',
                      MORTAL_ORACLE_PAUSE_FILE=str(output / 'pause.request'))
    sys.path.insert(0, str(source))
    import torch
    from mortal.online import pretrain_oracle_critic as pre
    from mortal.data.oracle_value import deterministic_game_id
    torch.set_num_threads(args.torch_threads)
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    torch.backends.cudnn.benchmark = bool(pre.config['control'].get('enable_cudnn_benchmark', True))
    tf32 = bool(pre.config['control'].get('allow_tf32', True))
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    cfg = dict(pre.oracle_pretrain_cfg())
    cfg.update(val_file_batch_size=args.file_batch, val_num_workers=0)
    files = pre.load_file_index(cfg['dev_file_index'])
    if args.files:
        files = files[:args.files]
    original_files = len(files)
    modulus, remainders = cfg['val_game_id_modulus'], cfg['val_game_id_remainders']
    if args.early_filter:
        files = [name for name in files if int(deterministic_game_id(name)) % modulus in remainders]
    keys = ('critic_arch', 'oracle_fusion_init', 'oracle_fusion_mode', 'oracle_fusion_hidden',
            'exact_zero_sum', 'value_loss_mode', 'value_head_hidden')
    brain, head = pre.build_models(torch.device('cpu'), **{key: cfg[key] for key in keys})
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    brain.load_state_dict(checkpoint['oracle_brain'])
    head.load_state_dict(checkpoint['value_net'])
    checkpoint_step = checkpoint['steps']
    resident_state = []
    if args.resident_optimizer:
        for state in checkpoint['optimizer']['state'].values():
            resident_state.extend(value.to('cuda') for value in state.values() if torch.is_tensor(value))
    del checkpoint
    brain.to('cuda')
    head.to('cuda')
    hashes = [hashlib.sha256() for _ in range(5)]
    counts, predictions, labels, game_ids = [], [], [], []
    def traced_batches():
        loader = pre.make_loader(pre.make_dataset(files, cfg, train=False), cfg, train=False)
        for batch in loader:
            keep = torch.tensor([int(game) % modulus in remainders for game in batch[4]])
            for hasher, field in zip(hashes, batch):
                hasher.update(field[keep].contiguous().numpy().tobytes())
            labels.append(batch[2][keep].clone())
            game_ids.append(batch[4][keep].clone())
            counts.append(int(keep.sum()))
            yield batch
    original_forward = pre.model_forward
    def forward(*values, **keywords):
        result = original_forward(*values, **keywords)
        predictions.append(result.detach().cpu())
        return result
    pre.model_forward = forward
    started = time.perf_counter()
    with Sampler(output, os.getpid()) as sampler:
        metrics = pre.evaluate_modes(brain, head, traced_batches(), torch.device('cuda'),
            enable_amp=bool(cfg.get('eval_enable_amp', False)), max_batches=0,
            input_modes=('true',), game_id_modulus=modulus, game_id_remainders=remainders,
            log_every_batches=64, include_cluster_records=True, label='resource-validation')['true']
    report = {'arguments': vars(args), 'checkpoint_step': checkpoint_step,
              'checkpoint_sha256': digest(args.checkpoint), 'source_config_sha256': digest(source_run / 'critic_config.toml'),
              'wall_s': time.perf_counter() - started, 'original_files': original_files,
              'loaded_files': len(files), 'samples': sum(counts),
              'resident_optimizer_bytes': sum(value.numel() * value.element_size() for value in resident_state),
              'ordered_input_field_sha256': [hasher.hexdigest() for hasher in hashes],
              'metrics': metrics, 'resources': sampler.report(),
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
              'scientific_use': False,
              'timing_scope': 'Includes input hashing and prediction capture; same instrumentation in each case.'}
    torch.save({'pred': torch.cat(predictions), 'target': torch.cat(labels), 'game_id': torch.cat(game_ids)},
               output / 'outputs.pth')
    dump(output / 'result.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'metrics'}), flush=True)


if __name__ == '__main__':
    main()
