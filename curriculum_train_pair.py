"""Resumable curriculum trainer; every phase uses a common frozen dataset."""
from pathlib import Path
import argparse,json,hashlib,math,time,random,signal
from collections import OrderedDict
import numpy as np
import torch
from full_fm_core import FullConditionalSolver,times_and_noise
from phase_metrics import metrics
ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--manifest',required=True);ap.add_argument('--mode',choices=['indi','fm','matched_endpoint'],required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--out',required=True);ap.add_argument('--resume');ap.add_argument('--stop-after',type=int);ap.add_argument('--transition');args=ap.parse_args();cfg=json.loads(Path(args.config).read_text());manifest=json.loads(Path(args.manifest).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert cfg['status']=='frozen_curriculum' and cfg['code_sha256']['curriculum_train_pair.py']==sha(__file__) and cfg['code_sha256']['full_fm_core.py']==sha(Path(__file__).with_name('full_fm_core.py'))
assert cfg['physics_gradient_audit_passed'] and cfg['dc_precision']=='float64'
assert manifest['status']=='frozen_after_license_hash_group_dedup_audit' and manifest['audit_passed']
phase=1;transition=None
if args.transition:
 transition=json.loads(Path(args.transition).read_text());assert transition['status']=='frozen_common_curriculum_transition' and transition['boundary_update']==5000;assert transition['config_sha256']==sha(args.config);assert transition['from_manifest_sha256']==cfg['manifest_sha256'] and transition['to_manifest_sha256']==sha(args.manifest);assert transition['all_four_arms_completed_boundary'] and transition['quality_used_for_data_choice'] is False;phase=2
else:assert cfg['manifest_sha256']==sha(args.manifest)
train=[x for x in manifest['records'] if x['split']=='train'];val=[x for x in manifest['records'] if x['split']=='val'];assert len(train)>=cfg['minimum_training_sources'] and len(val)==cfg['minimum_validation_sources'];assert not {x['source_group'] for x in train}&{x['source_group'] for x in val};assert all(x['license_verified'] for x in train+val)
assert hashlib.sha256(json.dumps([(x['source_id'],x['sha256']) for x in val],separators=(',',':')).encode()).hexdigest()==cfg['fixed_validation_identity_sha256']
if transition:
 assert {x['source_id'] for x in train}>={x for x in transition['initial_training_ids']};assert len(train)<=5000;byid={r['source_id']:r for r in train};assert all(byid[k]['sha256']==v for k,v in transition['initial_training_cache_sha256'].items())
out=Path(args.out);out.mkdir(exist_ok=args.resume is not None);stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True));torch.set_num_threads(4);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.cuda.set_per_process_memory_fraction(.85)
model=FullConditionalSolver(args.mode,T=cfg['T'],bridge_sigma=cfg['bridge_sigma'],dc_steps=cfg['training_dc_steps'],dc_precision=cfg.get('dc_precision','float32'));torch.nn.init.zeros_(model.raw.conv_out.weight);torch.nn.init.zeros_(model.raw.conv_out.bias);identity=hashlib.sha256()
for name,value in sorted(model.raw.state_dict().items()):identity.update(name.encode());identity.update(value.detach().cpu().contiguous().numpy().tobytes())
run_identity=dict(mode=args.mode,seed=args.seed,initial_backbone_sha256=identity.hexdigest(),manifest_sha256=cfg['manifest_sha256'],config_sha256=sha(args.config));ip=out/'run_identity.json'
if ip.exists():assert json.loads(ip.read_text())==run_identity
else:ip.write_text(json.dumps(run_identity,indent=2))
optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['peak_lr'],betas=(.95,.999),eps=1e-6,weight_decay=1e-5,foreach=False);start_step=0;best=-float('inf');epoch_orders={};verified=set();cpu_cache=OrderedDict();cpu_cache_bytes=0;cpu_cache_hits=0;cpu_cache_misses=0;cpu_cache_limit=int(cfg.get('cpu_cache_bytes',0));started=time.perf_counter();effective=cfg['effective_batch'];micro=cfg['microbatch'];assert effective%micro==0
def batch_from(records):
 global cpu_cache_bytes,cpu_cache_hits,cpu_cache_misses
 xs=[]
 for r in records:
  if r['path'] not in verified:assert sha(r['path'])==r['sha256'];verified.add(r['path'])
  key=r['path']
  if key in cpu_cache:x,size=cpu_cache.pop(key);cpu_cache[key]=(x,size);cpu_cache_hits+=1
  else:
   x=torch.load(key,map_location='cpu',weights_only=True);assert x['split']==r['split'];cpu_cache_misses+=1;size=sum(v.numel()*v.element_size() for v in x.values() if isinstance(v,torch.Tensor))
   if cpu_cache_limit and size<=cpu_cache_limit:
    while cpu_cache and cpu_cache_bytes+size>cpu_cache_limit:
     _,(_,old_size)=cpu_cache.popitem(last=False);cpu_cache_bytes-=old_size
    cpu_cache[key]=(x,size);cpu_cache_bytes+=size
  xs.append(x)
 return {k:torch.cat([x[k] for x in xs],0).cuda(non_blocking=False) for k in ('target','gt','hio','amp','support')}
def indices(step):
 ids=[]
 for i in range(effective):
  epoch,pos=divmod((step-(5000 if phase==2 else 0))*effective+i,len(train))
  if epoch not in epoch_orders:epoch_orders.clear();epoch_orders[epoch]=torch.randperm(len(train),generator=torch.Generator().manual_seed(args.seed+(1000003 if phase==2 else 0)+epoch*1009)).tolist()
  ids.append(epoch_orders[epoch][pos])
 return ids
def checkpoint(step,name,model_only=False):
 target=out/name;tmp=out/(name+'.tmp');payload=dict(model=model.state_dict(),step=step,best_validation=best,seed=args.seed,mode=args.mode,config=cfg,config_sha256=sha(args.config),manifest_sha256=sha(args.manifest),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),model_only=model_only,curriculum_phase=phase,transition_sha256=sha(args.transition) if args.transition else None)
 if not model_only:payload['optimizer']=optimizer.state_dict()
 torch.save(payload,tmp);tmp.replace(target)
if args.resume:
 ck=torch.load(args.resume,map_location='cpu',weights_only=True);assert ck['config_sha256']==sha(args.config) and ck['seed']==args.seed and ck['mode']==args.mode
 if ck['manifest_sha256']!=sha(args.manifest):
  assert transition and ck['step']==5000 and ck['manifest_sha256']==transition['from_manifest_sha256'];assert sha(args.resume)==transition['boundary_checkpoints'][str(args.seed)+'_'+args.mode]['sha256'];best=ck['best_validation']
 else:best=ck['best_validation']
 model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);start_step=ck['step'];torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);del ck
run_end=args.stop_after or cfg['updates'];assert start_step<=run_end<=cfg['updates'];assert phase==2 or run_end<=5000

def validation(step):
 global best
 model.eval();vals=[]
 with torch.no_grad():
  for r in val:
   b=batch_from([r]);gen=torch.Generator().manual_seed(23000+int(hashlib.sha256(r['source_id'].encode()).hexdigest()[:8],16));eps=torch.randn(b['target'].shape,generator=gen).cuda()
   with torch.autocast('cuda',dtype=torch.bfloat16):x=model.infer(b,nfe=4,noise=eps,dc_policy='per_stage5')
   # Frozen primary agrees with paperflip ambiguity; noGT shift for selection.
   y=b['gt'];mse=min(float((x-y).square().mean()),float((x.flip((-1,-2))-y).square().mean()));vals.append(-10*math.log10(max(mse,1e-12)))
 score=sum(vals)/len(vals);record=dict(step=step,mean_flip_psnr=score,source_scores=vals,source_ids=[r['source_id'] for r in val],curriculum_phase=phase,selection_nfe=4,dc_steps=20,test_read=False)
 with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 if score>best+1e-10:best=score;checkpoint(step,'best.pt')
 model.train()
log=(out/'training.jsonl').open('a');model.train();last_completed_step=start_step
if not args.resume:checkpoint(0,'last.pt');validation(0)
try:
 for step in range(start_step,run_end):
  if stop[0] or (out/'STOP').exists() or time.perf_counter()-started>cfg.get('max_run_seconds',25200):checkpoint(step,'last.pt');break
  lr=cfg['peak_lr']*min(1,(step+1)/cfg['warmup_updates'])*(.01+.99*.5*(1+math.cos(math.pi*max(0,step-cfg['warmup_updates'])/max(1,cfg['updates']-cfg['warmup_updates']))))
  for g in optimizer.param_groups:g['lr']=lr
  ids=indices(step);optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();tic=time.perf_counter();info=[]
  for off in range(0,effective,micro):
   b=batch_from([train[i] for i in ids[off:off+micro]]);t,eps=times_and_noise(args.seed,step,off,b['target'].shape,'cuda',cfg['T'])
   with torch.autocast('cuda',dtype=torch.bfloat16):loss,stats=model.training_loss(b,t,eps);loss=loss*(micro/effective)
   assert torch.isfinite(loss);loss.backward();info.append(stats)
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),2);assert torch.isfinite(norm);optimizer.step();last_completed_step=step+1;torch.cuda.synchronize();log.write(json.dumps(dict(update=step+1,source_ids=[train[i]['source_id'] for i in ids],lr=lr,gradient_norm=float(norm),seconds=time.perf_counter()-tic,losses=info,curriculum_phase=phase,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20))+'\n');log.flush()
  if (step+1)%cfg['validation_interval']==0:validation(step+1)
  if (step+1)%cfg['checkpoint_interval']==0:checkpoint(step+1,'last.pt')
  if step+1 in cfg.get('milestone_updates',[]):checkpoint(step+1,'stage_'+str(step+1)+'.pt',model_only=True)
 else:
  checkpoint(run_end,'last.pt')
  if run_end%cfg['validation_interval']!=0:validation(run_end)
finally:log.close();print(json.dumps(dict(finished_step=last_completed_step,requested_endpoint=run_end,curriculum_phase=phase,seconds=time.perf_counter()-started,mode=args.mode,seed=args.seed,test_read=False,cpu_cache_bytes=cpu_cache_bytes,cpu_cache_hits=cpu_cache_hits,cpu_cache_misses=cpu_cache_misses)))
