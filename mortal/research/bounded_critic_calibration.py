"""One finite critic-only calibration, using the production trainer and save clock."""
from __future__ import annotations
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

from mortal.core.artifacts import atomic_write_json, atomic_write_toml, atomic_torch_save, file_sha256
from mortal.core.update_clock import OptimizerUpdateClock
from mortal.online.calibration_bounds import calibration_limits, complete_mc_target
from mortal.research import critic_only_update_check as smoke
from mortal.research.frozen_actor_critic_probe import load_policy, load_verified_rollout, validate_critic_contract

SUCCESS_LIMIT = 256
PASS_LIMIT = 3
ATTEMPT_LIMIT = 318
SHUFFLE_SEED = 2026100402
SOURCE = Path(__file__).resolve().parents[2]


def build_config(base, actor_config, critic_state, args, root):
    controls = SimpleNamespace(**{**vars(args), "engineering_lr":1e-5, "batch_size":192, "enable_amp":True})
    cfg = smoke.build_config(base, actor_config, critic_state, controls, root)
    cfg['control'].update(state_file=str(root / 'latest.pth'), save_every=64,
                          allow_tf32=False, enable_cuda_prefetch=False)
    cfg['online'].update(max_successful_optimizer_steps=SUCCESS_LIMIT,
                         max_optimizer_attempts=ATTEMPT_LIMIT, max_replay_passes=PASS_LIMIT,
                         gae_inference_batch_size=512)
    cfg['value'].update(fixed_mc_targets=True, weight=1., zero_sum_weight=0.,
                        exact_zero_sum=True, independent_actor_lr_clock=False)
    cfg['expected_reward'] = {'enabled': False}
    cfg['optim'].update(betas=[.9,.999], eps=1e-8, weight_decay=0., max_grad_norm=0.)
    cfg['optim']['scheduler'].update(max_steps=ATTEMPT_LIMIT)
    cfg['repro'].update(seed=SHUFFLE_SEED, allow_cudnn_benchmark=False)
    calibration_limits(cfg)
    return cfg


def tensor_state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(__import__('torch').uint8).numpy().tobytes())
    return digest.hexdigest()


def assert_disjoint(frozen, trainable):
    def storages(models):
        return {(str(value.device), value.untyped_storage().data_ptr())
                for model in models for value in model.state_dict().values() if value.numel()}
    if storages(frozen) & storages(trainable):
        raise ValueError('frozen policy and trainable critic share tensor storage')


class FiniteDrain(smoke.OneShotDrain):
    def __call__(self):
        if self.calls >= PASS_LIMIT:
            raise RuntimeError('fourth replay pass forbidden')
        # Reuse the existing exact closed-directory and hash validation each pass.
        count = self.calls
        self.calls = 0
        value = super().__call__()
        self.calls = count + 1
        return value


def checked_trajectories(real_method, games, source, output, drain):
    import numpy as np
    index = json.loads((source / 'cache_index.json').read_text())
    cache = {(*group['key'], g['seat']): g for group in index['complete_groups'] for g in group['games']}
    originals = {str(Path(g['log_path']).resolve()): g for g in games}
    def wrapped(dataset, files, *, oracle_imputation_seed=None):
        if oracle_imputation_seed not in (None, smoke.IMPUTATION_SEED):
            raise ValueError('changed imputation seed')
        paths = list(files)
        trajectories = real_method(dataset, paths, oracle_imputation_seed=smoke.IMPUTATION_SEED)
        count = 0
        for count, (path, trajectory) in enumerate(zip(paths, trajectories, strict=True), 1):
            game = originals[str(Path(path).resolve())]
            key = (game['seed'], game['seed_key'], game['challenger_seat'])
            record = cache[key]; cp = source / record['path']
            if file_sha256(cp) != record['sha256']:
                raise ValueError('four-head cache changed')
            with gzip.open(cp, 'rt', encoding='utf-8') as handle:
                rows = [json.loads(line) for line in handle]
            if trajectory['player_id'] != key[2] or len(rows) != len(trajectory['obs']):
                raise ValueError('redecoded controlled identity/count changed')
            for field, row_key in [('actions','action'),('at_kyoku','at_kyoku'),
                                    ('decision_indices','decision_index')]:
                np.testing.assert_array_equal(trajectory[field], [r[row_key] for r in rows])
            np.testing.assert_array_equal(complete_mc_target(trajectory), [r['G_t'] for r in rows])
            atomic_write_json(output / 'replay_progress.json',
                              {'pass': drain.calls, 'last_seed': list(key), 'states': len(rows),
                               'four_head_fixed_mc_equal': True, 'unix': time.time()})
            yield trajectory
        if count != len(paths):
            raise ValueError('native trajectory count changed')
    return wrapped


class CalibrationObservation:
    def __init__(self, torch, actor_state, critic_state, fixed, initial_logits, root):
        self.torch, self.actor_state, self.critic_state = torch, actor_state, critic_state
        self.fixed, self.initial_logits, self.root = fixed, initial_logits, root
        self.models = {}
        self.frozen_initial = {}
        self.events, self.steps = [], []
        self.offered = self.rejected = self.publications = 0

    def logits(self, actor, policy):
        obs, masks = (x.to(next(actor.parameters()).device) for x in self.fixed)
        with self.torch.inference_mode(), self.torch.autocast(obs.device.type, enabled=False):
            return policy.logits(actor(obs), masks).detach().cpu()

    def check_frozen(self, *, full):
        for name, model in self.models.items():
            if name in ('oracle_brain','value_net'):
                continue
            if any(p.grad is not None for p in model.parameters()):
                raise ValueError(f'frozen gradients present: {name}')
            if name in ('mortal','policy_net','Old_mortal','Old_policy_net'):
                if any(m.training for m in model.modules()):
                    raise ValueError(f'frozen actor left eval mode: {name}')
            if full:
                expected = self.actor_state['mortal' if name=='Old_mortal' else 'policy_net'
                                            if name=='Old_policy_net' else name] if name in (
                                                'mortal','policy_net','Old_mortal','Old_policy_net') else self.frozen_initial[name]
                smoke.assert_state_equal(model.state_dict(), expected, self.torch, name)
        if full:
            if not self.torch.equal(self.initial_logits, self.logits(self.models['mortal'],self.models['policy_net'])):
                raise ValueError('frozen S70 logits changed')

    def __call__(self, event, state):
        torch = self.torch
        if event == 'before_critic_load':
            names = ('mortal','policy_net','Old_mortal','Old_policy_net','aux_net','opponent_aux_net',
                     'danger_aux_net','tile_eff_net','furo_regret_net','hand_value_regret_net',
                     'oracle_brain','value_net')
            self.models = {n:state[n] for n in names if state.get(n) is not None}
            for name, model in self.models.items():
                if name not in ('oracle_brain','value_net','mortal','policy_net','Old_mortal','Old_policy_net'):
                    self.frozen_initial[name] = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if state['optimizer'].state or state['update_clock'].attempts or state['update_clock'].progress:
                raise ValueError('calibration optimizer and clocks must be fresh')
            self.check_frozen(full=True)
        elif event == 'after_critic_load':
            self.check_frozen(full=True)
            assert_disjoint([m for n,m in self.models.items() if n not in ('oracle_brain','value_net')],
                            [self.models['oracle_brain'], self.models['value_net']])
            for name in ('oracle_brain','value_net'):
                smoke.assert_state_equal(self.models[name].state_dict(),self.critic_state[name],torch,name)
            torch.manual_seed(SHUFFLE_SEED)
            __import__('numpy').random.seed(SHUFFLE_SEED)
        elif event == 'batch_offered':
            self.offered += 1
            if state['obs'].shape[0] != 192 or state['v_target'].shape != (192,4):
                raise ValueError('changed physical batch or four-head target')
            if not torch.equal(state['v_target'],state['v_target'].round()):
                raise ValueError('raw MC target altered before batching')
            return
        elif event == 'drift_rejected':
            self.rejected += 1
            atomic_write_json(self.root/'anomaly.json',{'event':event,'offered':self.offered})
            raise RuntimeError('frozen old/current policy drift rejected a batch')
        elif event == 'before_step':
            self.check_frozen(full=False)
            if state['policy_step_active'] is not False:
                raise ValueError('actor/auxiliary update gate enabled')
            target = state['value_target']
            if target.shape != (192,4) or not torch.equal(target,target.round()):
                raise ValueError('value target normalized/clipped/changed')
            for key in ('aux_loss_val','opp_loss_val','danger_loss_val','exp_reward_loss_val',
                        'tile_eff_loss_val','furo_regret_loss_val','hand_value_regret_loss_val'):
                if float(state[key].detach().cpu()) != 0.:
                    raise ValueError(f'non-value loss active: {key}')
            loss, value_loss = float(state['loss'].detach().cpu()),float(state['value_loss_val'].detach().cpu())
            if not __import__('math').isfinite(loss) or loss != value_loss:
                raise ValueError('loss differs from sole raw all-player MSE')
            rates = [float(g['lr']) for g in state['optimizer'].param_groups]
            if any(abs(lr-1e-5)>1e-14 for lr in rates):
                raise ValueError(f'calibration LR changed: {rates}')
            self.pending = {'loss':loss,'lr':rates,'offered_batches':self.offered,
                            'approx_kl':float(state['drift']['approx_kl']),
                            'clip_fraction':float(state['drift']['clip_fraction'])}
            return
        elif event == 'after_step':
            clock=state['update_clock']
            self.check_frozen(full=clock.attempts%64==0 or clock.successes==SUCCESS_LIMIT)
            self.steps.append({**self.pending,'succeeded':state['step_succeeded'],
                               'clock':clock.state_dict(),'unix':time.time()})
            atomic_write_json(self.root/'observed_updates.json',self.steps)
            if clock.attempts%16==0 or clock.successes==SUCCESS_LIMIT:
                print(json.dumps({'actual_clock':clock.state_dict(),'loss':self.pending['loss'],
                                  'offered':self.offered,'unix':time.time()}),flush=True)
            return
        else:
            raise ValueError(f'unknown observation: {event}')
        self.events.append({'event':event,'actor_state_sha256':tensor_state_hash(self.models['mortal'].state_dict()),
                            'policy_state_sha256':tensor_state_hash(self.models['policy_net'].state_dict()),
                            'frozen_state_buffers_logits_equal':True,'unix':time.time()})
        atomic_write_json(self.root/'load_integrity.json',self.events)

    def submit(self, actor, policy, **kwargs):
        self.check_frozen(full=False)
        version=self.publications; self.publications+=1
        return version


def run(args, root):
    if 'mortal.config' in sys.modules:
        raise RuntimeError('calibration must run in a fresh process')
    if not os.environ.get('MORTAL_RUN_ID') or not os.environ.get('MORTAL_RUN_DEADLINE_UTC'):
        raise RuntimeError('independent existing deadline supervisor required')
    os.environ['RAYON_NUM_THREADS']='4'
    import numpy as np
    import torch
    import libriichi
    from mortal.core.toml_utils import load_toml_file
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    files={name:Path(getattr(args,name)).resolve() for name in ('config','actor','critic','opponent')}
    hashes={name:file_sha256(p) for name,p in files.items()}
    for name in ('actor','critic','opponent'):
        if hashes[name] != getattr(args,name+'_sha256'):raise ValueError(f'{name} hash mismatch')
    if file_sha256(libriichi.__file__) != args.native_sha256:raise ValueError('native hash mismatch')
    source=Path(args.reuse_rollout_dir).resolve()
    if file_sha256(source/'cache_index.json') != args.cache_index_sha256:raise ValueError('cache index mismatch')
    previous=json.loads((source/'provenance.json').read_text())
    for old,new in [('actor','actor'),('opponent','opponent'),('reference','critic')]:
        if previous['weights_and_config'][old]['sha256']!=hashes[new]:raise ValueError('B128 role mismatch')
    actor_state=torch.load(files['actor'],map_location='cpu',weights_only=True,mmap=True)
    critic_state=torch.load(files['critic'],map_location='cpu',weights_only=True,mmap=True)
    if actor_state['steps']!=390000 or critic_state['steps']!=250000:raise ValueError('S70/selected250k internal step mismatch')
    pre=validate_critic_contract(critic_state,version=4,pts=[2,1,0,-3])
    modes=[g.get('train_mode') for g in critic_state['optimizer']['param_groups']]
    if not modes or any(x is not False for x in modes):raise ValueError('selected ScheduleFree eval readout not proven')
    cfg=build_config(load_toml_file(files['config']),actor_state['config'],critic_state,args,root)
    from datetime import datetime
    hard_deadline=datetime.fromisoformat(os.environ['MORTAL_RUN_DEADLINE_UTC']).timestamp()
    cfg['online']['calibration_stop_unix']=hard_deadline-540
    atomic_write_toml(root/'effective_config.toml',cfg)
    os.environ['MORTAL_CFG']=str(root/'effective_config.toml')
    from mortal.config import config
    from mortal.core import common
    from mortal.data.dataloader import FileDatasetsIter
    from mortal.eval.paired_1v3 import load_games, duplicate_sets
    from mortal.online import train_online
    native=Path(libriichi.__file__)
    provenance={'schema':1,'status':'running','started_unix':time.time(),
        'arguments':{**vars(args),'games':128,'imputation_seed':smoke.IMPUTATION_SEED,
                     'torch_threads':1,'rayon_threads':4},
        'weights_and_config':{n:{'path':str(p),'sha256':hashes[n]} for n,p in files.items()},
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
        'source_status':subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE,text=True).strip(),
        'source_hashes':{n:file_sha256(SOURCE/n) for n in smoke.SOURCE_FILES},
        'native_sha256':file_sha256(native),
        'native_extension_sha256':{str(p):file_sha256(p) for p in sorted(set(native.parent.glob('*.pyd'))|set(native.parent.glob('*.so')))},
        'torch_version':torch.__version__,'numpy_version':np.__version__,
        'actor_explore_rate':1.,'opponent_explore_rate':0.,'actor_agari_guard':False,
        'opponent_agari_guard':True,'search_enabled':False,'actor_oracle_guiding':False,'probe_pts':[2,1,0,-3],
        'fixed_MC':True,'fresh_AdamW':True,'schedulefree_eval_modes':modes,'shuffle_seed':SHUFFLE_SEED,
        'success_limit':SUCCESS_LIMIT,'pass_limit':PASS_LIMIT,'attempt_limit':ATTEMPT_LIMIT}
    games,reuse=load_verified_rollout(source,provenance,load_games=load_games,duplicate_sets=duplicate_sets)
    provenance['rollout_reuse']=reuse
    if provenance['source_status']:raise ValueError('dirty source')
    atomic_write_json(root/'provenance.json',provenance)
    protected={p:hashes[n] for n,p in files.items()}
    protected.update({Path(g['log_path']).resolve():g['sha256'] for g in games})
    drain=FiniteDrain(source/'games',games)
    first=sorted(g['log_path'] for g in games)[:1]
    dataset=FileDatasetsIter(version=4,file_list=first,pts=[2,1,0,-3],oracle=True,
                            player_names=['trainee'],value_target_mode='all_players',value_reward_source='score_rank')
    trajectory=next(dataset.iter_game_trajectories(first,oracle_imputation_seed=smoke.IMPUTATION_SEED))
    fixed=(torch.from_numpy(trajectory['obs'][:2].copy()).float(),torch.from_numpy(trajectory['masks'][:2].copy()).bool())
    del trajectory,dataset
    actor,policy,_,_=load_policy(files['actor'],torch)
    actor.to(args.device);policy.to(args.device)
    with torch.inference_mode(),torch.autocast(torch.device(args.device).type,enabled=False):
        initial_logits=policy.logits(actor(fixed[0].to(args.device)),fixed[1].to(args.device)).cpu()
    initial_hashes={n:tensor_state_hash(m.state_dict()) for n,m in [('mortal',actor),('policy_net',policy)]}
    atomic_write_json(root/'actor_before_any_critic_load.json',initial_hashes)
    del actor,policy
    observer=CalibrationObservation(torch,actor_state,critic_state,fixed,initial_logits,root)
    exit_code=None
    try:
        with patch.object(common,'drain',drain),patch.object(common,'submit_param',observer.submit), \
             patch.object(FileDatasetsIter,'iter_game_trajectories',checked_trajectories(
                 FileDatasetsIter.iter_game_trajectories,games,source,root,drain)):
            try:train_online.train(calibration_observer=observer)
            except SystemExit as exc:
                exit_code=exc.code
                if exit_code not in (86,87):raise
        latest_path=Path(config['control']['state_file'])
        latest=torch.load(latest_path,map_location='cpu',weights_only=True,mmap=True)
        clock=OptimizerUpdateClock.from_checkpoint(latest,opt_step_every=1)
        if clock.attempts!=len(observer.steps) or clock.successes!=sum(x['succeeded'] for x in observer.steps):
            raise ValueError('real observed updates differ from saved clock')
        if clock.legacy_attempt_offset or clock.inherited_progress_offset or clock.successes>256 or clock.attempts>318:
            raise ValueError('calibration clock bound or freshness violated')
        complete=clock.successes==256 and exit_code==86
        if clock.successes==256 and not complete:raise ValueError('exact endpoint exited with inconsistent status')
        observer.check_frozen(full=True)
        for n in ('mortal','policy_net'):
            smoke.assert_state_equal(latest[n],actor_state[n],torch,'saved '+n)
        for n in observer.frozen_initial:
            smoke.assert_state_equal(latest[n],observer.frozen_initial[n],torch,'saved '+n)
        changed={}
        for n in ('oracle_brain','value_net'):
            if any(not bool(torch.isfinite(v).all()) for v in latest[n].values()):raise ValueError('nonfinite saved critic')
            changed[n]=sum(not torch.equal(v,critic_state[n][k]) for k,v in latest[n].items())
        if complete and not all(changed.values()):raise ValueError('candidate critic did not actually change')
        # Strictly reload the saved actor on the same FP32 device and inputs.
        reloaded_actor,reloaded_policy,_,_=load_policy(latest_path,torch)
        reloaded_actor.to(args.device);reloaded_policy.to(args.device)
        if not torch.equal(initial_logits,observer.logits(reloaded_actor,reloaded_policy)):
            raise ValueError('saved actor logits changed')
        del reloaded_actor,reloaded_policy
        result={'status':'complete' if complete else 'incomplete','planned_candidate':complete,
                'trainer_exit_code':exit_code,'optimizer_update_clock':clock.state_dict(),
                'drain_calls':drain.calls,'offered_batches':observer.offered,'drift_rejections':observer.rejected,
                'critic_changed_tensors':changed,'actor_parameters_buffers_logits_equal':True,
                'actor_state_sha256':tensor_state_hash(latest['mortal']),
                'policy_state_sha256':tensor_state_hash(latest['policy_net']),
                'latest_sha256':file_sha256(latest_path),'finished_unix':time.time(),
                'automatic_resume':False,'holdout_allowed':complete}
        if complete:
            # Weights-only inference artifact; preserve the true 256-success identity.
            export={'config':deepcopy(latest['config']),'steps':256,
                    'oracle_critic_pretrain':deepcopy(critic_state['oracle_critic_pretrain']),
                    'oracle_brain':latest['oracle_brain'],'value_net':latest['value_net'],
                    'optimizer_update_clock':clock.state_dict(),'source_latest_sha256':result['latest_sha256'],
                    'reference_sha256':hashes['critic'],'artifact_role':'calibration256_eval_weights_only'}
            export['oracle_critic_pretrain']['max_steps']=256
            atomic_torch_save(export,root/'candidate_eval.pth')
            result['candidate_eval_sha256']=file_sha256(root/'candidate_eval.pth')
        atomic_write_json(root/'result.json',result)
        print(json.dumps(result),flush=True)
    finally:
        smoke.verify_hashes(protected)
        atomic_write_json(root/'input_immutability.json',{'unchanged':True,'hashes':{str(p):h for p,h in protected.items()}})
    provenance.update(status=result['status'],finished_unix=time.time(),result_sha256=file_sha256(root/'result.json'))
    atomic_write_json(root/'provenance.json',provenance)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','actor','critic','opponent','reuse-rollout-dir','output-dir',
                 'actor-sha256','critic-sha256','opponent-sha256','native-sha256','cache-index-sha256'):
        parser.add_argument('--'+name,required=True)
    for name in ('seed-start','seed-key','sampling-seed'):
        parser.add_argument('--'+name,type=int,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args(argv)
    root=smoke.reserve_check_output(args.output_dir,args.reuse_rollout_dir)
    atomic_write_json(root/'request.json',vars(args))
    try:run(args,root)
    except BaseException as exc:
        atomic_write_json(root/'failure.json',{'type':type(exc).__name__,'error':str(exc),'unix':time.time()})
        raise


if __name__=='__main__':
    main()
