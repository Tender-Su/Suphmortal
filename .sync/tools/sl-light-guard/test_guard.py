import ast, importlib.util, pathlib, unittest
P=pathlib.Path(__file__).with_name('backend_guard.py')
spec=importlib.util.spec_from_file_location('guard',P);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
class GuardTests(unittest.TestCase):
 def test_reserves(self):
  b={'minimum_available_ram_bytes':2*1024**3,'minimum_global_vram_free_mib':1024}
  good={'ram_available_bytes':3*1024**3,'gpu_free_mib':2000,'commit_available_bytes':2*1024**3}
  self.assertFalse(g.reserve_breached(good,b))
  for k,v in [('ram_available_bytes',2*1024**3-1),('gpu_free_mib',1023),('commit_available_bytes',1024**3-1)]:
   self.assertTrue(g.reserve_breached(dict(good,**{k:v}),b))
 def test_no_midrun_audits(self):
  tree=ast.parse(P.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
  loop=next(n for n in ast.walk(main) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='iteration')
  self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='audit' for n in ast.walk(loop)))
  self.assertEqual(sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='audit' for n in ast.walk(main)),1)
 def test_budget_and_counters_are_not_b7000_specific(self):
  s=P.read_text()
  for obsolete in ('update-5000','skipped-2','delta<=2000','range(1200)','max(0,7000-update)'):
   self.assertNotIn(obsolete,s)
  self.assertIn("bounds['target_update']",s)
 def test_seal_contract(self):
  s=P.read_text();self.assertIn("'sealed.json' if bounds.get('validation_only')",s)
  self.assertIn("delta==bounds['maximum_new_successful_updates']",s)
  self.assertIn("'4060' in v[1]",s)
if __name__=='__main__':unittest.main()
