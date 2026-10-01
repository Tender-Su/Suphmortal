from copy import deepcopy
import unittest
from mortal.supervised.continuation import backend_protocol, apply_backend_config, relocate_config, rebind_saved_state
from mortal.tests.test_sl_continuation import saved_fixture
from scripts.continue_sl_phase import parser

class BackendProtocolTests(unittest.TestCase):
    def test_inherited_default(self):
        self.assertIsNone(backend_protocol('inherit', None, ''))
        for threads, reason in [(1, ''), (None, 'changed')]:
            with self.assertRaises(ValueError): backend_protocol('inherit', threads, reason)

    def test_canonical_validation(self):
        for mode, threads, reason in [('unknown',1,'x'), ('fast',True,'x'), ('fast',3,'x'), ('fast',1,'')]:
            with self.assertRaises(ValueError): backend_protocol(mode,threads,reason)
        fast=backend_protocol('fast',1,'measured 2.25x cached throughput')
        strict=backend_protocol('strict',2,'reference')
        self.assertFalse(fast['deterministic'])
        self.assertTrue(strict['deterministic'])
        tampered=dict(fast,allow_tf32=False)
        with self.assertRaises(ValueError): apply_backend_config({'control':{}},tampered)

    def test_only_declared_control_changes(self):
        source=saved_fixture(); original=deepcopy(source)
        backend=backend_protocol('fast',1,'sweep winner')
        config=relocate_config(source['config'],'/tmp/new','new',{'checkpoint_id':'parent'},'commit','old',runtime_sha256='digest',backend=backend)
        migrated=rebind_saved_state(source,config,'new',backend=backend)
        for key in source:
            if key not in ('config','run_provenance','curriculum_probe','checkpoint_id'):
                self.assertEqual(source[key],migrated[key])
        self.assertEqual(source,original)
        self.assertEqual(source['curriculum_probe']['dataset'],migrated['curriculum_probe']['dataset'])
        self.assertEqual(source['curriculum_probe']['rng'],migrated['curriculum_probe']['rng'])
        self.assertEqual(config['supervised']['run_provenance']['backend_protocols'],[backend])
        for key in ('allow_tf32','enable_cudnn_benchmark'): self.assertTrue(config['control'][key])
        for section,key,value in [('supervised','lr',.123),('control','opt_step_every',99)]:
            bad=deepcopy(config);bad[section][key]=value
            with self.assertRaises(ValueError): rebind_saved_state(source,bad,'new',backend=backend)
        with self.assertRaises(ValueError): rebind_saved_state(source,config,'new')

    def test_cli(self):
        args=parser().parse_args(['prepare','--source-run','s','--checkpoint','p','--directory','d','--source-commit','c',
            '--until-update','50000','--save-every-updates','500','--trend-every-updates','5000','--full-every-updates','10000',
            '--trend-recent-games','128','--trend-old-games','64','--trend-seed','1','--save-every-seconds','300',
            '--backend','fast','--torch-threads','1','--backend-change-reason','sweep winner'])
        self.assertEqual(backend_protocol(args.backend,args.torch_threads,args.backend_change_reason)['torch_threads'],1)

if __name__=='__main__': unittest.main()
