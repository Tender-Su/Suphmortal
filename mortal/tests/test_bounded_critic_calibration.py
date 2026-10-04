"""Focused CPU contracts for the bounded critic-only adapter and paired scorer."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from mortal.core.update_clock import OptimizerUpdateClock, observed_scaler_step
from mortal.online.calibration_bounds import calibration_limits, calibration_limit_reason, complete_mc_target
from mortal.research.bounded_critic_calibration import (
    FiniteDrain, CalibrationObservation, assert_disjoint, build_config, tensor_state_hash)
from mortal.research.paired_critic_calibration import paired_block, input_fingerprint
from mortal.research.frozen_actor_critic_probe import parse_args
from mortal.tests.test_critic_only_update_check import ConfigurationTests


class BoundedConfigTests(ConfigurationTests):
    def test_fresh_single_setting_and_no_legacy_objectives(self):
        cfg=build_config(self.base,self.actor,self.critic,self.args,Path('/new'))
        self.assertEqual(cfg['control']['batch_size'],192)
        self.assertEqual(cfg['control']['save_every'],64)
        self.assertEqual(cfg['online']['gae_inference_batch_size'],512)
        self.assertEqual(calibration_limits(cfg),(3,318))
        self.assertEqual(cfg['online']['max_successful_optimizer_steps'],256)
        self.assertTrue(cfg['value']['fixed_mc_targets'])
        self.assertEqual(cfg['value']['weight'],1.)
        self.assertEqual(cfg['value']['zero_sum_weight'],0.)
        self.assertEqual(cfg['optim']['weight_decay'],0.)
        self.assertEqual(cfg['optim']['scheduler'],
                         {'init':1e-5,'peak':1e-5,'final':1e-5,'warm_up_steps':0,'max_steps':318})
        self.assertEqual(cfg['aux'],self.actor['aux'])
        self.assertFalse(cfg['expected_reward']['enabled'])


class LimitsTests(unittest.TestCase):
    def valid(self):
        return {'online':{'max_replay_passes':3,'max_optimizer_attempts':318},
                'value':{'enabled':True,'critic_only':True,'oracle_critic':True,'target_mode':'all_players',
                         'reward_source':'score_rank','fixed_mc_targets':True},
                'policy':{'gae_enabled':True,'gae_gamma':1.,'gae_lambda':1.,'online_action_scope':'all'}}
    def test_default_is_inactive_and_explicit_bounds(self):
        self.assertEqual(calibration_limits({}),(0,0))
        self.assertEqual(calibration_limits(self.valid()),(3,318))
        self.assertIsNone(calibration_limit_reason((3,318),completed_passes=2,attempts=317))
        self.assertEqual(calibration_limit_reason((3,318),completed_passes=3,attempts=250),'replay_pass_limit')
        self.assertEqual(calibration_limit_reason((3,318),completed_passes=2,attempts=318),'optimizer_attempt_limit')
    def test_reject_target_changes_and_actor_updates(self):
        for section,key,value in [('value','critic_only',False),('value','target_mode','current_player'),
                                   ('policy','gae_lambda',.95),('policy','online_action_scope','discard')]:
            c=self.valid();c[section][key]=value
            with self.assertRaises(ValueError):calibration_limits(c)
        c=self.valid();c['online']['importance_sampling']={'enabled':True}
        with self.assertRaises(ValueError):calibration_limits(c)
    def test_full_game_raw_mc_includes_skipped_kyoku_and_terminal_once(self):
        t={'obs':np.zeros((3,1)), 'at_kyoku':np.array([0,0,2]),
           'kyoku_value_target':np.array([[1,-1,0,0],[2,0,-2,0],[-3,0,0,3]],dtype=np.float32)}
        actual=complete_mc_target(t)
        np.testing.assert_array_equal(actual,[[0,-1,-2,3],[0,-1,-2,3],[-3,0,0,3]])
    def test_production_saves_after_last_pass_before_next_drain(self):
        text=(Path(__file__).resolve().parents[1]/'online/train_online.py').read_text()
        order=['completed_replay_passes = 0','train_epoch()','completed_replay_passes += 1',
               'stop_at_calibration_boundary(completed_replay_passes)']
        start=text.index(order[0])
        for piece in order:
            found=text.index(piece,start);start=found+len(piece)
        self.assertIn('persist_live_training_state(reward_target_metadata_dict=dict(reward_target_metadata))',
                      text[text.index('def stop_at_calibration_boundary'):text.index('def stop_after_checkpoint_if_requested')])
        self.assertIn("v_tgt = complete_mc_target(traj)",text)
        self.assertNotIn('clip',text[text.index('value_loss_val = nn.functional.mse_loss'):text.index('value_loss_val = nn.functional.mse_loss')+70])
    def test_production_cutoff_saves_before_new_optimizer_clock(self):
        import ast
        text=(Path(__file__).resolve().parents[1]/'online/train_online.py').read_text()
        tree=ast.parse(text)
        nodes={node.name:node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef)}
        saved=[];flushed=[]
        env={'config':{'online':{'calibration_stop_unix':660}},'calibration_bounds':(3,318),
             'training_stop_requested':lambda:False,'_time':SimpleNamespace(time=lambda:660.),
             'persist_live_training_state':lambda **kw:saved.append(kw),'reward_target_metadata':{},
             'writer':SimpleNamespace(flush=lambda:flushed.append(True)),
             'logging':SimpleNamespace(info=lambda *a:None),'steps':17,
             'sys':SimpleNamespace(exit=lambda code:(_ for _ in ()).throw(SystemExit(code))),
             'ONLINE_STOP_REQUEST_EXIT_CODE':87}
        module=ast.Module(body=[nodes['stop_after_checkpoint_if_requested']],type_ignores=[])
        exec(compile(ast.fix_missing_locations(module),'real_production_cutoff','exec'),env)
        with self.assertRaises(SystemExit) as stop:env['stop_after_checkpoint_if_requested']()
        self.assertEqual(stop.exception.code,87);self.assertEqual(len(saved),1);self.assertEqual(flushed,[True])
        marker=text.index('# A forward begun before cutoff')
        self.assertLess(text.index('stop_after_checkpoint_if_requested()',marker),text.index('steps += 1',marker))

    def test_three_readonly_drains_and_no_fourth(self):
        from mortal.core.artifacts import file_sha256
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'one';p.write_text('same immutable replay')
            drain=FiniteDrain(d,[{'log_path':str(p),'sha256':file_sha256(p)}])
            for _ in range(3):self.assertEqual(drain(),str(Path(d).resolve()))
            with self.assertRaises(RuntimeError):drain()
            self.assertEqual(p.read_text(),'same immutable replay')
    def test_remainder_arithmetic(self):
        self.assertEqual(divmod(20365,192),(106,13))
        self.assertEqual((20365//192)*3,318)
    def test_u64_seed_no_implicit_truncation(self):
        args=parse_args(['--config','c','--actor','a','--opponent','o','--output-dir','x',
            '--reference','r','--reference-sha256','a'*64,'--reference-steps','250000',
            '--candidate','q','--candidate-sha256','b'*64,'--candidate-steps','256',
            '--seed-start','20261004100','--seed-key','2026100401','--sampling-seed','202610040256'])
        self.assertEqual(args.seed_start,20261004100)
        self.assertGreater(args.seed_start,2**32)
        from mortal.core.repro import resolve_train_seed_start,resolve_train_key
        config={'repro':{'train_seed_start':args.seed_start,'train_key':args.seed_key}}
        self.assertEqual(resolve_train_seed_start(config),args.seed_start)
        self.assertEqual(resolve_train_key(config),args.seed_key)


class FreezeTests(unittest.TestCase):
    def test_detect_shared_storage_and_buffers(self):
        a=torch.nn.BatchNorm1d(4).eval();b=deepcopy(a)
        assert_disjoint([a],[b])
        b.weight=a.weight
        with self.assertRaises(ValueError):assert_disjoint([a],[b])
        h=tensor_state_hash(a.state_dict());a.running_mean[0]=1
        self.assertNotEqual(tensor_state_hash(a.state_dict()),h)
    def test_real_adam_update_counts_and_preserves_actor_old_buffers_logits(self):
        class Policy(torch.nn.Linear):
            def logits(self,phi,masks):return self(phi).masked_fill(~masks,-torch.inf)
        actor=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.BatchNorm1d(4)).eval()
        policy=Policy(4,4).eval();critic=torch.nn.Linear(4,4);value=torch.nn.Linear(4,4)
        models={'mortal':actor,'policy_net':policy,'Old_mortal':deepcopy(actor),
                'Old_policy_net':deepcopy(policy),'oracle_brain':critic,'value_net':value}
        a={n:{k:v.detach().clone() for k,v in m.state_dict().items()} for n,m in models.items() if n in ('mortal','policy_net')}
        c={n:{k:v.detach().clone() for k,v in m.state_dict().items()} for n,m in models.items() if n in ('oracle_brain','value_net')}
        x=torch.randn(192,4);mask=torch.ones((192,4),dtype=torch.bool)
        with torch.inference_mode():initial=policy.logits(actor(x[:2]),mask[:2])
        optimizer=torch.optim.AdamW([p for m in (actor,policy,critic,value) for p in m.parameters()],lr=1e-5,weight_decay=0.)
        clock=OptimizerUpdateClock()
        with tempfile.TemporaryDirectory() as d:
            obs=CalibrationObservation(torch,a,c,(x[:2],mask[:2]),initial,Path(d))
            local={**models,'optimizer':optimizer,'update_clock':clock}
            obs('before_critic_load',local);obs('after_critic_load',local)
            y=torch.zeros_like(x);pred=value(critic(x));loss=torch.nn.functional.mse_loss(pred,y);loss.backward()
            local.update(policy_step_active=False,value_target=y,v_target=y,obs=x,loss=loss,
                         value_loss_val=loss,drift={'approx_kl':torch.tensor(0.),'clip_fraction':torch.tensor(0.)})
            for n in ('aux_loss_val','opp_loss_val','danger_loss_val','exp_reward_loss_val',
                      'tile_eff_loss_val','furo_regret_loss_val','hand_value_regret_loss_val'):local[n]=torch.tensor(0.)
            obs('batch_offered',local);obs('before_step',local)
            local['step_succeeded']=observed_scaler_step(torch.amp.GradScaler('cpu',enabled=False),optimizer,clock)
            obs('after_step',local);obs.check_frozen(full=True)
            self.assertEqual(clock.successes,1);self.assertEqual(clock.attempts,1)
            self.assertNotEqual(tensor_state_hash(critic.state_dict()),tensor_state_hash(c['oracle_brain']))


class PairTests(unittest.TestCase):
    def games(self,n=4):
        return [{'seed':100+i//4,'seed_key':1,'challenger_seat':i%4} for i in range(n)]
    def test_opponent_gain_cannot_hide_p0_regression(self):
        y=[np.zeros((2,4)) for _ in range(4)]
        a=[np.tile([0,2,2,2],(2,1)) for _ in range(4)]
        b=[np.tile([1,0,0,0],(2,1)) for _ in range(4)]
        r=paired_block(self.games(),y,a,b,[np.ones(2,dtype=bool)]*4,replicates=1000,seed=1)
        self.assertEqual(r['paired_delta']['estimate'][2],1.)
        self.assertEqual(r['paired_delta']['estimate'][3],-2.75)
    def test_ratio_of_sums_and_a_a_exact_zero(self):
        y=[np.zeros((1 if i<4 else 3,4)) for i in range(8)]
        a=[np.zeros_like(x) for x in y];b=[np.full_like(x,1 if i<4 else 2) for i,x in enumerate(y)]
        masks=[np.ones(len(x),dtype=bool) for x in y]
        r=paired_block(self.games(8),y,a,b,masks,replicates=1000,seed=1)
        self.assertEqual(r['paired_delta']['estimate'][2],3.25)
        aa=paired_block(self.games(8),y,b,b,masks,replicates=1000,seed=1)
        self.assertEqual(aa['paired_delta']['estimate'],[0.]*7)
        self.assertEqual(aa['paired_delta']['ci95_high'],[0.]*7)
    def test_empty_stratum_is_reported_and_mismatches_rejected(self):
        y=[np.zeros((2,4)) for _ in range(4)]
        r=paired_block(self.games(),y,y,y,[np.zeros(2,dtype=bool)]*4,replicates=1000,seed=1)
        self.assertEqual(r['states'],0)
        with self.assertRaises(ValueError):
            paired_block(self.games(),y,y,y[:-1],[np.ones(2,dtype=bool)]*4,replicates=1000,seed=1)
    def test_shared_input_fingerprint_detects_hidden_or_target_change(self):
        t={k:np.zeros((2,4)) for k in ('obs','invisible_obs','masks')}
        t.update(actions=np.array([0,1]),at_kyoku=np.array([0,0]),decision_indices=np.arange(2))
        y=np.zeros((2,4));first=input_fingerprint(t,y)
        t['invisible_obs'][0,0]=1
        self.assertNotEqual(first,input_fingerprint(t,y))
