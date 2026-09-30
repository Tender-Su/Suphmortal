"""Read-only, opt-in CUDA regression probe; run only on an idle training GPU."""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    os.environ['MORTAL_CFG'] = str(Path(args.config).resolve())
    os.environ.setdefault('RAYON_NUM_THREADS', '2')

    import torch
    from torch.utils.data import DataLoader
    from mortal.config import config
    from mortal.core.model import Brain, CategoricalPolicy
    from mortal.data.dataloader import SupervisedFileDatasetsIter
    from mortal.supervised.train_supervised import (
        forward_validation_brain, safe_default_collate,
    )

    torch.set_num_threads(2)
    torch.manual_seed(17)
    cfg = config['supervised']
    device = torch.device(config['control']['device'])
    state = torch.load(cfg['state_file'], map_location='cpu', weights_only=False)
    model = Brain(version=config['control']['version'], Norm='GN', **config['resnet'])
    policy = CategoricalPolicy()
    model.load_state_dict(state['mortal'])
    policy.load_state_dict(state['policy_net'])
    model.to(device)
    policy.to(device)
    index = torch.load(cfg['file_index'], map_location='cpu', weights_only=False)
    dataset = SupervisedFileDatasetsIter(
        version=config['control']['version'],
        file_list=list(index['monitor_recent_files'][:16]),
        file_batch_size=1, reserve_ratio=0, shuffle_files=False,
        rayon_num_threads=2, emit_game_id=True,
    )
    loader = iter(DataLoader(
        dataset, batch_size=cfg['batch_size'], num_workers=0,
        collate_fn=safe_default_collate,
    ))
    batch = next(loader)
    obs, actions, masks = batch[:3]
    assert len(obs) == cfg['batch_size']
    actions = actions.to(device)
    masks = masks.to(device)
    # Preserve two training-sized observations across validation, like current
    # and prefetched training batches. Optimizer tensors are resident too.
    resident = [obs.to(device=device, dtype=torch.float32).clone() for _ in range(2)]
    optimizer_tensors = [
        value.to(device)
        for item in state['optimizer']['state'].values()
        for value in item.values() if torch.is_tensor(value)
    ]
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32 = config['control'].get('allow_tf32', True)
    torch.backends.cudnn.allow_tf32 = config['control'].get('allow_tf32', True)
    amp = config['control']['enable_amp']

    def training_backward():
        model.train()
        policy.train()
        with torch.autocast(device.type, enabled=amp):
            logits = policy.logits(model(resident[0]), masks)
            loss = torch.nn.functional.cross_entropy(logits, actions)
        loss.backward()
        assert torch.isfinite(loss)
        model.zero_grad(set_to_none=True)
        policy.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return float(loss.detach())

    before = training_backward()
    model.eval()
    policy.eval()
    results = []
    outputs = []
    for size in (0, 256):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.autocast(device.type, enabled=amp):
            phi = model(resident[0]) if size == 0 else forward_validation_brain(
                model, obs, device=device, microbatch_size=size,
            )
            logits = policy.logits(phi, masks)
            loss = torch.nn.functional.cross_entropy(logits, actions)
            outputs.append(logits.float().cpu())
            result = {
                'microbatch_size': size, 'policy_loss': float(loss),
                'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
            }
            del phi, logits, loss
        torch.cuda.synchronize()
        results.append(result)
    legal = masks.cpu()
    delta = (outputs[0][legal] - outputs[1][legal]).abs()
    agreement = float((outputs[0].argmax(-1) == outputs[1].argmax(-1)).float().mean())
    after = training_backward()
    report = {
        'checkpoint_step': state['steps'], 'samples': len(obs),
        'optimizer_tensors_resident': len(optimizer_tensors),
        'resident_training_batches': len(resident),
        'train_before_loss': before, 'train_after_loss': after,
        'validation': results, 'max_legal_logit_delta': float(delta.max()),
        'mean_legal_logit_delta': float(delta.mean()), 'action_agreement': agreement,
    }
    print(json.dumps(report), flush=True)
    assert abs(results[0]['policy_loss'] - results[1]['policy_loss']) < 2e-4
    assert agreement >= 0.999
    assert results[1]['peak_allocated_mib'] < results[0]['peak_allocated_mib']
    assert abs(before - after) < 2e-4


if __name__ == '__main__':
    main()

