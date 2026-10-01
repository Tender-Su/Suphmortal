"""Finite resource/lease guard and read-only evidence collection; never restarts."""
from pathlib import Path
import ctypes,datetime as dt,hashlib,json,math,os,pstats,re,subprocess,sys,time,traceback
from ctypes import wintypes as w
ROOT=Path(__file__).resolve().parent;NEW=Path(os.environ.get('SL_EXISTING_RUN',str(ROOT/'run'))).resolve();LEASE=ROOT/'production_lease'
def now():return dt.datetime.now(dt.timezone.utc)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def put(name,v):
 p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8');os.replace(t,p)
def event(v):
 with (ROOT/'stage_events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'utc':now().isoformat(),**v},allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def resources():
 class Mem(ctypes.Structure):
  _fields_=[('length',w.DWORD),('load',w.DWORD)]+[(n,ctypes.c_ulonglong) for n in ('total','available','total_page','available_page','total_virtual','available_virtual','extended')]
 m=Mem();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
 p=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,encoding='utf-8',timeout=8,creationflags=subprocess.CREATE_NO_WINDOW)
 assert p.returncode==0,p.stderr;v=[x.strip() for x in p.stdout.strip().split(',')]
 assert v[0]=='GPU-375f4a45-4075-4aae-ac24-b40d55f2ed3c' and '4060' in v[1]
 return {'utc':now().isoformat(),'ram_available_bytes':m.available,'ram_total_bytes':m.total,'commit_limit_bytes':m.total_page,'commit_available_bytes':m.available_page,'commit_used_bytes':m.total_page-m.available_page,'paging_activity_measured':False,'gpu_used_mib':int(v[2]),'gpu_free_mib':int(v[3]),'gpu_utilization_percent':int(v[4])}
def reserve_breached(sample,bounds):
 return (sample['ram_available_bytes']<bounds['minimum_available_ram_bytes'] or sample['gpu_free_mib']<bounds['minimum_global_vram_free_mib'] or sample['commit_available_bytes']<1024**3)
def stop(reason,state):
 if not state.get('stop_requested_utc'):
  (LEASE/'STOP').touch(exist_ok=True);state.update(stop_requested_utc=now().isoformat(),stop_reason=reason);event({'event':'cooperative_stop_requested','reason':reason})
def tail(path):
 if not path.exists():return ''
 with path.open('rb') as f:f.seek(max(0,path.stat().st_size-300000));return f.read().decode('utf-8','replace')
def audit():
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 p=subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'checkpoint_metadata.py'),str(NEW/'state_file.pth')],capture_output=True,text=True,encoding='utf-8',errors='replace',env=env,timeout=60,creationflags=subprocess.CREATE_NO_WINDOW)
 assert p.returncode==0,p.stderr[-2500:];return json.loads(p.stdout)
def profile_summary():
 path=ROOT/'runtime.pstats'
 if not path.exists():return {'status':'unavailable','reason':'profile not emitted before trainer exit'}
 stats=pstats.Stats(str(path));functions=[];totals={}
 for fn,values in stats.stats.items():
  filename,line,name=fn;cc,nc,own,cumulative,callers=values
  norm=filename.replace('\\','/');base=norm.rsplit('/',1)[-1]
  selected=(base=='ordered_preparation.py' or (base=='train_supervised.py' and name in ('train','forward_loss','apply_optimizer_step','save_latest_checkpoint','build_state','evaluate')) or (base=='artifacts.py' and name=='atomic_torch_save') or (base=='_tensor.py' and name=='backward'))
  if not selected:continue
  rows=[]
  for caller,timing in callers.items():
   cf,cl,cn=caller;vals=list(timing) if isinstance(timing,tuple) else [timing]
   rows.append({'file':cf.replace('\\','/').rsplit('/',1)[-1],'line':cl,'name':cn,'profile_values':vals})
  functions.append({'file':base,'line':line,'name':name,'primitive_calls':cc,'calls':nc,'own_seconds':own,'cumulative_seconds':cumulative,'callers':rows})
  if base=='ordered_preparation.py' and name=='__iter__':totals['main_thread_preparation_and_input_wait_seconds']=totals.get('main_thread_preparation_and_input_wait_seconds',0)+cumulative
  if base=='train_supervised.py' and name=='evaluate':totals['validation_function_seconds']=totals.get('validation_function_seconds',0)+cumulative
  if base=='artifacts.py' and name=='atomic_torch_save':totals['checkpoint_serialization_seconds']=totals.get('checkpoint_serialization_seconds',0)+cumulative
  if base=='train_supervised.py' and name=='save_latest_checkpoint':totals['latest_save_including_build_seconds']=cumulative
  if base=='train_supervised.py' and name=='apply_optimizer_step':totals['training_optimizer_seconds']=cumulative
  if base=='_tensor.py' and name=='backward':totals['training_backward_seconds']=cumulative
  if base=='train_supervised.py' and name=='forward_loss':
   totals['training_forward_loss_seconds']=sum(float(x[-1]) for (cf,cl,cn),x in callers.items() if cn=='train' and isinstance(x,tuple))
 return {'status':'available','instrument':'standard-library cProfile around unchanged frozen entry','timing_clock':'wall/perf-counter main-thread call timings; GPU asynchronous calls are not kernel busy time','scope':'profile overhead included in end-to-end throughput; preparation includes main-thread native-pool wait/row assembly, not overlapping worker compute; nested save components overlap and must not be added twice','total_profiled_seconds':stats.total_tt,'totals':totals,'functions':functions}
def main():
 bounds=read(ROOT/'bounds.json');initial=read(ROOT/'initial_checkpoint.json');end=dt.datetime.fromisoformat(bounds['hard_deadline_utc']);soft=dt.datetime.fromisoformat(bounds['compute_stop_utc'])
 state={'status':'monitoring','guard_pid':os.getpid(),'run_id':os.environ.get('MORTAL_RUN_ID'),'started_utc':now().isoformat(),'hard_deadline_utc':end.isoformat(),'compute_stop_utc':soft.isoformat(),'identity':initial['identity'],'source_commit':initial['source_commit'],'resource_samples':0,'ram_available_min_bytes':None,'gpu_free_min_mib':None,'gpu_used_peak_mib':0,'new_successful_updates':0,'new_skipped_updates':0,'no_automatic_retry_or_protocol_change':True}
 assert state['run_id'] and os.environ['MORTAL_RUN_DEADLINE_UTC']==end.isoformat()
 log=LEASE/'sl512_fast.stderr.log';seen_steps=initial['steps'];saved_mtime=(NEW/'state_file.pth').stat().st_mtime_ns;first=None;last_audit=0;resource_errors=0;seen_saves=set()
 try:
  for iteration in range(1200):
   if now()>=end:break
   assert read(LEASE/'manifest.json')['run_id']==state['run_id']
   result=LEASE/'sl512_fast.result.json'
   if result.exists():
    state['trainer_result']=read(result);break
   if now()>=soft:stop('declared compute cutoff; reserve final grace for save/exit',state)
   try:
    sample=resources();resource_errors=0;state['resource_samples']+=1;state['latest_resources']=sample
    state['ram_available_min_bytes']=sample['ram_available_bytes'] if state['ram_available_min_bytes'] is None else min(state['ram_available_min_bytes'],sample['ram_available_bytes'])
    state['gpu_free_min_mib']=sample['gpu_free_mib'] if state['gpu_free_min_mib'] is None else min(state['gpu_free_min_mib'],sample['gpu_free_mib']);state['gpu_used_peak_mib']=max(state['gpu_used_peak_mib'],sample['gpu_used_mib'])
    with (ROOT/'resources.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(sample)+'\n')
    if reserve_breached(sample,bounds):stop('RAM, commit or global VRAM reserve breached',state)
   except Exception as exc:
    resource_errors+=1;state['resource_probe_error']=repr(exc)
    if resource_errors>=3:stop('three consecutive resource probe failures',state)
   text=tail(log)
   skips=re.findall(r'GradScaler skipped optimizer update[^\r\n]*skipped=(\d+)\)',text);skipped=state.get('skipped_optimizer_updates',initial['skipped_optimizer_steps'])
   if skips:skipped=max(skipped,int(skips[-1]))
   state['skipped_optimizer_updates']=skipped
   rows=re.findall(r'epoch=\d+ step=([\d,]+) loss=([\d.eE+-]+)',text)
   if rows:
    steps=int(rows[-1][0].replace(',',''));loss=float(rows[-1][1]);assert math.isfinite(loss)
    if steps>seen_steps:
     update=steps//2-skipped;record={'utc':now().isoformat(),'microsteps':steps,'optimizer_updates':update,'new_successful_updates':update-5000,'skipped_optimizer_updates':skipped,'loss':loss}
     event({'event':'training_progress',**record});state.update(latest_training=record,new_successful_updates=update-5000,new_skipped_updates=skipped-2);seen_steps=steps
     if first is None:first=record;state['first_real_update']=record;put('first_real_update.json',record)
     if update>first['optimizer_updates']:
      seconds=(now()-dt.datetime.fromisoformat(first['utc'])).total_seconds();rate=(update-first['optimizer_updates'])/seconds
      state['logged_training_updates_per_second']=rate;state['training_only_eta_utc']=(now()+dt.timedelta(seconds=max(0,7000-update)/rate)).isoformat();state['eta_scope']='Short startup log interval; excludes remaining full validation, capped by hard deadline.'
   for line in text.splitlines():
    if 'saved latest ' in line and 'optimizer_steps=' in line and line not in seen_saves:
     seen_saves.add(line);event({'event':'checkpoint_save_completed','log':line[-650:]})
   if any(token in text for token in ('OutOfMemoryError','CUDA out of memory','FloatingPointError')):stop('trainer error detected; preserve branch without restart',state)
   # Resource sampling must never block on Torch/checkpoint/index loads.
   state['checkpoint_file_mtime_ns']=(NEW/'state_file.pth').stat().st_mtime_ns
   state['checkpoint_file_bytes']=(NEW/'state_file.pth').stat().st_size
   state['midrun_checkpoint_audits_enabled']=False
   state['updated_utc']=now().isoformat();put('guard_latest.json',state)
   time.sleep(max(0,min(10,(end-now()).total_seconds())))
  state['trainer_finished_observed_utc']=now().isoformat()
  assert 'trainer_result' in state,'hard deadline reached before trainer exit could be audited'
  final=audit();put('final_checkpoint.json',final)
  delta=final['optimizer_steps']-initial['optimizer_steps'];skipped_delta=final['skipped_optimizer_steps']-initial['skipped_optimizer_steps']
  assert final['identity']==initial['identity'] and final['source_commit']==initial['source_commit'] and final['microbatch']==512 and final['accumulation']==2
  assert final['allow_tf32'] and final['enable_cudnn_benchmark'] and 0<=delta<=2000 and skipped_delta>=0
  assert final['auxiliary_optimizer_steps']-initial['auxiliary_optimizer_steps']==delta
  assert final['adam_step_min']-initial['adam_step_min']==delta and final['adam_step_max']-initial['adam_step_max']==delta
  assert final['scheduler']['last_epoch']-initial['scheduler']['last_epoch']==delta
  assert final['consumed_decisions']-initial['consumed_decisions']==(delta+skipped_delta)*1024
  assert final['steps']-initial['steps']==2*(delta+skipped_delta)
  unchanged=all(sha(p)==h for p,h in read(ROOT/'protected_inputs.json').items());assert unchanged,'protected original or prepared input changed'
  profile=profile_summary();put('timing_profile.json',profile)
  completion_path=NEW/('sealed.json' if bounds.get('validation_only') else 'completed.json')
  completed=read(completion_path) if completion_path.exists() else None
  validations=[read(p) for p in sorted((NEW/'full').glob('update_*.json'))] if (NEW/'full').exists() else []
  eval_seconds=sum(v.get('evaluation_seconds',0) for v in validations)
  wall=(dt.datetime.fromisoformat(state['trainer_result']['finished_utc'])-dt.datetime.fromisoformat(bounds['lease_start_utc'])).total_seconds()
  success=state['trainer_result']['returncode']==0 and delta==bounds['maximum_new_successful_updates'] and completed is not None and completed['optimizer_updates']==7000 and any(v['optimizer_updates']==7000 for v in validations)
  state.update(status='completed_verified' if success else 'stopped_or_failed_state_preserved',new_successful_updates=delta,new_skipped_updates=skipped_delta,new_nonfinite_batches=final['nonfinite_batches']-initial['nonfinite_batches'],final_checkpoint_sha256=final['checkpoint_sha256'],full_state_clocks_and_consumption_verified=True,protected_inputs_unchanged=unchanged,end_to_end_wall_seconds=wall,end_to_end_updates_per_second=delta/wall,validation_seconds=eval_seconds,phase_elapsed_delta_seconds=final['elapsed_seconds']-initial['elapsed_seconds'],completed=completed,profile_available=profile['status']=='available')
  return 0 if success or state['trainer_result']['returncode']==75 else 1
 except BaseException as exc:
  stop('guard failure: '+repr(exc),state);state.update(status='guard_failed',error=repr(exc),traceback=traceback.format_exc()[-3500:]);return 1
 finally:
  state['finished_utc']=now().isoformat();put('guard_latest.json',state);put('final_summary.json',state)
if __name__=='__main__':raise SystemExit(main())

