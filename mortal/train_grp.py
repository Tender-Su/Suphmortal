import prelude

import random
import torch
import logging
from math import inf
from os import path
from glob import glob
from datetime import datetime
from torch import optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.tensorboard import SummaryWriter
from cpu_affinity import maybe_configure_process_affinity
from model import GRP
from libriichi.dataset import Grp
from common import tqdm
from config import config


def resolve_dtype(name):
    return {
        'float32': torch.float32,
        'float64': torch.float64,
    }[name]

class GrpFileDatasetsIter(IterableDataset):
    def __init__(
        self,
        file_list,
        file_batch_size = 50,
        cycle = False,
        dtype = torch.float64,
    ):
        super().__init__()
        self.file_list = file_list
        self.file_batch_size = file_batch_size
        self.cycle = cycle
        self.dtype = dtype
        self.buffer = []

    def build_iter(self, file_list):
        while True:
            random.shuffle(file_list)
            for start_idx in range(0, len(file_list), self.file_batch_size):
                self.populate_buffer(file_list[start_idx:start_idx + self.file_batch_size])
                buffer_size = len(self.buffer)
                for i in random.sample(range(buffer_size), buffer_size):
                    yield self.buffer[i]
                self.buffer.clear()
            if not self.cycle:
                break

    def populate_buffer(self, file_list):
        # Check if we are loading preprocessed .pt files or raw .json.gz files
        if len(file_list) > 0 and file_list[0].endswith('.pt'):
            for file in file_list:
                data = torch.load(file, weights_only=False)
                for item in data:
                    feature = item[0]
                    rank_by_player = item[1]
                    for i in range(feature.shape[0]):
                        self.buffer.append((
                            feature[:i + 1].to(self.dtype),
                            rank_by_player
                        ))
        else:
            data = Grp.load_gz_log_files(file_list)
            for game in data:
                feature = game.take_feature()
                rank_by_player = game.take_rank_by_player()

                for i in range(feature.shape[0]):
                    inputs_seq = torch.as_tensor(feature[:i + 1], dtype=self.dtype)
                    self.buffer.append((
                        inputs_seq,
                        rank_by_player,
                    ))

    def __iter__(self):
        worker_info = get_worker_info()
        file_list = list(self.file_list)
        if worker_info is not None:
            per_worker = (len(file_list) + worker_info.num_workers - 1) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(file_list))
            file_list = file_list[start:end]
        return self.build_iter(file_list)

def collate(batch):
    inputs = []
    lengths = []
    rank_by_players = []
    for inputs_seq, rank_by_player in batch:
        inputs.append(inputs_seq)
        lengths.append(len(inputs_seq))
        rank_by_players.append(rank_by_player)

    lengths = torch.tensor(lengths)
    rank_by_players = torch.tensor(rank_by_players, dtype=torch.int64, pin_memory=True)

    padded = pad_sequence(inputs, batch_first=True)
    packed_inputs = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
    packed_inputs = packed_inputs.pin_memory()

    return packed_inputs, rank_by_players


def load_or_build_file_lists(file_index, train_globs, val_globs):
    index_signature = {
        'train_globs': list(train_globs),
        'val_globs': list(val_globs),
    }

    if path.exists(file_index):
        index = torch.load(file_index, weights_only=True)
        if index.get('signature') == index_signature:
            return index['train_file_list'], index['val_file_list']
        logging.info('grp file index changed, rebuilding...')
    else:
        logging.info('building file index...')

    train_file_list = []
    val_file_list = []
    for pat in train_globs:
        train_file_list.extend(glob(pat, recursive=True))
    for pat in val_globs:
        val_file_list.extend(glob(pat, recursive=True))
    train_file_list.sort(reverse=True)
    val_file_list.sort(reverse=True)
    torch.save({
        'signature': index_signature,
        'train_file_list': train_file_list,
        'val_file_list': val_file_list,
    }, file_index)
    return train_file_list, val_file_list


def make_data_loader(file_list, file_batch_size, batch_size, num_workers, cycle, dtype, device):
    file_data = GrpFileDatasetsIter(
        file_list=file_list,
        file_batch_size=file_batch_size,
        cycle=cycle,
        dtype=dtype,
    )
    loader_kwargs = {
        'dataset': file_data,
        'batch_size': batch_size,
        'drop_last': cycle,
        'num_workers': num_workers,
        'collate_fn': collate,
    }
    if num_workers > 0:
        loader_kwargs['worker_init_fn'] = grp_worker_init_fn
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = config['grp']['dataset'].get('prefetch_factor', 4)
    return DataLoader(
        **loader_kwargs,
    )


def grp_worker_init_fn(*args, **kwargs):
    maybe_configure_process_affinity(log=False, context='grp dataloader worker')


def run_validation(grp, val_data_loader, device, val_steps, dtype):
    grp.eval()
    val_loss = 0.
    val_acc = 0.
    total_batches = 0
    pb = tqdm(total=val_steps if val_steps > 0 else None, desc='VAL')

    with torch.inference_mode():
        for idx, (inputs, rank_by_players) in enumerate(val_data_loader):
            if val_steps > 0 and idx == val_steps:
                break
            inputs = inputs.to(dtype=dtype, device=device, non_blocking=device.type == 'cuda')
            rank_by_players = rank_by_players.to(dtype=torch.int64, device=device, non_blocking=device.type == 'cuda')

            logits = grp.forward_packed(inputs)
            labels = grp.get_label(rank_by_players)
            loss = F.cross_entropy(logits, labels)

            val_loss += loss.item()
            val_acc += (logits.argmax(-1) == labels).to(torch.float64).mean().item()
            total_batches += 1
            pb.update(1)

    pb.close()
    grp.train()

    if total_batches == 0:
        raise RuntimeError('validation produced zero batches')

    return {
        'val_loss': val_loss / total_batches,
        'val_acc': val_acc / total_batches,
        'val_batches': total_batches,
    }


def save_state(
    state_path,
    grp,
    optimizer,
    scheduler,
    steps,
    epoch,
    best_val_loss,
    best_val_acc,
    best_loss_epoch,
    best_acc_epoch,
    patience_counter,
    num_lr_reductions,
):
    state = {
        'model': grp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'steps': steps,
        'epoch': epoch,
        'timestamp': datetime.now().timestamp(),
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'best_loss_epoch': best_loss_epoch,
        'best_acc_epoch': best_acc_epoch,
        'patience_counter': patience_counter,
        'num_lr_reductions': num_lr_reductions,
    }
    torch.save(state, state_path)
    return state

def train():
    cfg = config['grp']
    batch_size = cfg['control']['batch_size']
    val_steps = cfg['control']['val_steps']
    max_epochs = cfg['control'].get('max_epochs', 0)
    min_epochs = cfg['control'].get('min_epochs', 1)
    patience = cfg['control'].get('early_stopping_patience', 0)
    min_delta = cfg['control'].get('early_stopping_min_delta', 0.)
    min_lr_reductions = cfg['control'].get('early_stopping_min_lr_reductions', 0)
    dtype = resolve_dtype(cfg['network'].get('dtype', 'float64'))

    device = torch.device(cfg['control']['device'])
    torch.backends.cudnn.benchmark = cfg['control']['enable_cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32 = cfg['control'].get('allow_tf32', True)
    torch.backends.cudnn.allow_tf32 = cfg['control'].get('allow_tf32', True)
    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    grp = GRP(**cfg['network']).to(dtype=dtype, device=device)
    optimizer = optim.AdamW(grp.parameters())
    scheduler_cfg = cfg['optim'].get('scheduler', {})
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_cfg.get('factor', 0.5),
        patience=scheduler_cfg.get('patience', 2),
        threshold=scheduler_cfg.get('threshold', 0.0005),
        cooldown=scheduler_cfg.get('cooldown', 1),
        min_lr=scheduler_cfg.get('min_lr', 1e-6),
    )

    state_file = cfg['state_file']
    latest_state_file = cfg.get('latest_state_file', state_file)
    best_loss_state_file = cfg.get('best_loss_state_file', state_file)
    best_acc_state_file = cfg.get('best_acc_state_file', state_file)
    resume_state_file = latest_state_file if path.exists(latest_state_file) else state_file
    if path.exists(resume_state_file):
        state = torch.load(resume_state_file, weights_only=True, map_location=device)
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        grp.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        if state.get('scheduler') is not None:
            scheduler.load_state_dict(state['scheduler'])
        steps = state['steps']
        start_epoch = state.get('epoch', 0)
        best_val_loss = state.get('best_val_loss', inf)
        best_val_acc = state.get('best_val_acc', 0.)
        best_loss_epoch = state.get('best_loss_epoch', 0)
        best_acc_epoch = state.get('best_acc_epoch', 0)
        patience_counter = state.get('patience_counter', 0)
        num_lr_reductions = state.get('num_lr_reductions', 0)
    else:
        steps = 0
        start_epoch = 0
        best_val_loss = inf
        best_val_acc = 0.
        best_loss_epoch = 0
        best_acc_epoch = 0
        patience_counter = 0
        num_lr_reductions = 0

    initial_lr = cfg['optim']['lr']
    if start_epoch == 0:
        optimizer.param_groups[0]['lr'] = initial_lr

    file_index = cfg['dataset']['file_index']
    train_globs = cfg['dataset']['train_globs']
    val_globs = cfg['dataset']['val_globs']
    train_file_list, val_file_list = load_or_build_file_lists(file_index, train_globs, val_globs)
    if not train_file_list:
        raise FileNotFoundError(f'no GRP training files matched: {train_globs}')
    if not val_file_list:
        raise FileNotFoundError(f'no GRP validation files matched: {val_globs}')

    num_workers = cfg['dataset'].get('num_workers', 6)

    writer = SummaryWriter(cfg['control']['tensorboard_dir'])

    logging.info(f'train file list size: {len(train_file_list):,}')
    logging.info(f'val file list size: {len(val_file_list):,}')

    if max_epochs <= 0:
        raise ValueError('grp.control.max_epochs must be set to a positive integer')

    for epoch in range(start_epoch + 1, max_epochs + 1):
        train_loss = 0.
        train_acc = 0.
        train_batches = 0
        train_data_loader = make_data_loader(
            file_list=train_file_list,
            file_batch_size=cfg['dataset']['file_batch_size'],
            batch_size=batch_size,
            num_workers=num_workers,
            cycle=False,
            dtype=dtype,
            device=device,
        )

        pb = tqdm(desc=f'TRAIN {epoch}/{max_epochs}')
        for inputs, rank_by_players in train_data_loader:
            inputs = inputs.to(dtype=dtype, device=device, non_blocking=device.type == 'cuda')
            rank_by_players = rank_by_players.to(dtype=torch.int64, device=device, non_blocking=device.type == 'cuda')

            logits = grp.forward_packed(inputs)
            labels = grp.get_label(rank_by_players)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (logits.argmax(-1) == labels).to(torch.float64).mean().item()
            train_batches += 1
            steps += 1
            pb.update(1)
        pb.close()

        if train_batches == 0:
            raise RuntimeError('training produced zero batches')

        val_data_loader = make_data_loader(
            file_list=val_file_list,
            file_batch_size=cfg['dataset']['file_batch_size'],
            batch_size=batch_size,
            num_workers=num_workers,
            cycle=False,
            dtype=dtype,
            device=device,
        )
        val_metrics = run_validation(grp, val_data_loader, device, val_steps, dtype)

        train_loss /= train_batches
        train_acc /= train_batches
        val_loss = val_metrics['val_loss']
        val_acc = val_metrics['val_acc']
        current_lr = optimizer.param_groups[0]['lr']

        writer.add_scalars('loss', {
            'train': train_loss,
            'val': val_loss,
        }, steps)
        writer.add_scalars('acc', {
            'train': train_acc,
            'val': val_acc,
        }, steps)
        writer.add_scalar('epoch', epoch, steps)
        writer.add_scalar('lr', current_lr, steps)
        writer.flush()

        logging.info(
            f'epoch {epoch:02d}/{max_epochs:02d} '
            f'steps={steps:,} '
            f'train_loss={train_loss:.4f} train_acc={train_acc:.4f} '
            f'val_loss={val_loss:.4f} val_acc={val_acc:.4f}'
        )

        improved_loss = val_loss < best_val_loss - min_delta
        improved_acc = val_acc > best_val_acc

        if improved_loss:
            best_val_loss = val_loss
            best_loss_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            logging.info(
                f'no loss improvement: best_epoch={best_loss_epoch} best_val_loss={best_val_loss:.4f} '
                f'patience={patience_counter}/{patience}'
            )

        if improved_acc:
            best_val_acc = val_acc
            best_acc_epoch = epoch

        if improved_loss:
            save_state(
                best_loss_state_file,
                grp,
                optimizer,
                scheduler,
                steps,
                epoch,
                best_val_loss,
                best_val_acc,
                best_loss_epoch,
                best_acc_epoch,
                patience_counter,
                num_lr_reductions,
            )
            logging.info(f'new best model saved: epoch={epoch} val_loss={val_loss:.4f} val_acc={val_acc:.4f}')

        if improved_acc:
            save_state(
                best_acc_state_file,
                grp,
                optimizer,
                scheduler,
                steps,
                epoch,
                best_val_loss,
                best_val_acc,
                best_loss_epoch,
                best_acc_epoch,
                patience_counter,
                num_lr_reductions,
            )
            logging.info(f'new best-acc model saved: epoch={epoch} val_acc={val_acc:.4f} val_loss={val_loss:.4f}')

        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < current_lr:
            num_lr_reductions += 1
            logging.info(
                f'lr reduced: {current_lr:.6g} -> {new_lr:.6g} '
                f'(reductions={num_lr_reductions})'
            )

        save_state(
            latest_state_file,
            grp,
            optimizer,
            scheduler,
            steps,
            epoch,
            best_val_loss,
            best_val_acc,
            best_loss_epoch,
            best_acc_epoch,
            patience_counter,
            num_lr_reductions,
        )

        if (
            patience > 0
            and epoch >= min_epochs
            and patience_counter >= patience
            and num_lr_reductions >= min_lr_reductions
        ):
            logging.info(
                f'early stopping triggered at epoch {epoch}; '
                f'best_loss_epoch={best_loss_epoch} best_val_loss={best_val_loss:.4f} '
                f'best_acc_epoch={best_acc_epoch} best_val_acc={best_val_acc:.4f}'
            )
            break

    writer.close()

if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        pass
