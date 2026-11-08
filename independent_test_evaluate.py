"""One fixed heldout evaluation of every preregistered checkpoint and sampler."""
from pathlib import Path
import argparse,json,hashlib,time,statistics,math,gc,sys,types
import torch
from PIL import Image
from full_fm_core import FullConditionalSolver,fft2d,zero_padding_twice,crop_center_half
from phase_metrics import metrics
from upstream_adapter import get_algorithm

ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);ap.add_argument('--lock',required=True);ap.add_argument('--train-manifest',required=True);ap.add_argument('--manifest');ap.add_argument('--audit');ap.add_argument('--out',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
out=Path(a.out);out.mkdir(exist_ok=False);protocol=json.loads(Path(a.protocol).read_text());proof=json.loads(Path(a.lock).read_text());train=json.loads(Path(a.train_manifest).read_text());trainrows=[r for r in train['records'] if r['split']=='train'];torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.85)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
started=time.monotonic();cache={};rows=[];summaries=[];first8=set();qual=out/'qualitative';qual.mkdir();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(a.protocol)==proof['protocol_sha256'] and sha(a.train_manifest)==proof['stage2_manifest_sha256']
assert sha(Path(__file__).parent/'full_fm_core.py')==protocol['core_sha256']

def load(meta):
 if meta['source_id'] not in cache:
  assert sha(meta['path'])==meta['sha256'];cache[meta['source_id']]=torch.load(meta['path'],map_location='cpu',weights_only=True)
 return {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in cache[meta['source_id']].items()}
def noise(sid):
 seed=23000+int(hashlib.sha256(sid.encode()).hexdigest()[:8],16)
 return torch.randn((1,1,256,256),generator=torch.Generator().manual_seed(seed)).cuda()
def quality(x,b):
 q=metrics(x,b['gt'],b['amp']);mse=min(float((x-b['gt']).square().mean()),float((x.flip((-1,-2))-b['gt']).square().mean()))
 q.update(flip_mse=mse,flip_psnr=-10*math.log10(max(mse,1e-12)),failed=False);return q
def failed(reason):return dict(raw_mse=1.,raw_psnr=0.,aligned_mse=1.,aligned_psnr=0.,flip_mse=1.,flip_psnr=0.,amp_residual=None,transform=None,failed=True,failure_reason=reason)
def timed(fn,n):
 times=[];x=None
 for _ in range(n):torch.cuda.synchronize();tic=time.perf_counter();x=fn();torch.cuda.synchronize();times.append(time.perf_counter()-tic)
 return x,statistics.median(times),times
def picture(sid,name,x):
 if sid not in first8:return
 folder=qual/sid.replace(':','_');folder.mkdir(exist_ok=True)
 Image.fromarray((x.detach().float().cpu().squeeze().clamp(0,1).numpy()*255).round().astype('uint8')).save(folder/(name+'.png'))
@torch.no_grad()
def hio(b,branch,k,dc):
 if branch=='reference':initial=b['hio'][:,:1]
 elif branch=='mean':initial=b['hio'].mean(1,keepdim=True)
 else:
  residue=(fft2d(zero_padding_twice(b['hio'])).abs()-b['amp']).square().mean((-1,-2));j=int(residue.argmin());initial=b['hio'][:,j:j+1]
 if k:
  z,_=get_algorithm('HIO')(b['amp']*255,b['support'],k,initial_value=zero_padding_twice(initial*255),use_tqdm=False);D=crop_center_half(z).clamp(0,255)/255
 else:D=initial
 if dc:
  fake=types.SimpleNamespace(dc_steps=5,dc_precision='float64',lam=torch.logspace(-.64,-.10,32).cuda());D=FullConditionalSolver.dc(fake,D,b['amp'],b['support'],torch.tensor([32],device='cuda'),5)
 return D
def model_from(key):
 meta=proof['checkpoints'][key];assert sha(meta['path'])==meta['sha256'];ck=torch.load(meta['path'],map_location='cpu',weights_only=True)
 assert ck['step']==meta['step'] and ck['config_sha256']==protocol['config_sha256']
 model=FullConditionalSolver(ck['mode'],T=32,bridge_sigma=3/255,dc_precision='float64');model.load_state_dict(ck['model']);model.eval();del ck;return model
def forward(model,b,sid,nfe,policy):
 # This function deliberately cannot hand GT or training targets to the solver.
 inputs={k:b[k] for k in ('amp','hio','support')}
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):return model.infer(inputs,nfe,noise(sid),policy)

if a.smoke:
 checks=[]
 for mode in ('indi','fm'):
  model=model_from('2027_'+mode+'_best')
  for meta in trainrows[:4]:
   b=load(meta);outputs={}
   for nfe in (1,4,8):
    for policy in ('per_stage5','fixed_total40'):
     x=forward(model,b,meta['source_id'],nfe,policy);assert torch.isfinite(x).all();outputs[(nfe,policy)]=x
   err=float((outputs[(8,'per_stage5')]-outputs[(8,'fixed_total40')]).abs().max());assert err<1e-6
   checks.append(dict(mode=mode,training_source=meta['source_id'],all_six_variants_finite=True,nfe8_equivalence_max_error=err))
  del model;gc.collect();torch.cuda.empty_cache()
 report=dict(status='post_training_train_only_sampler_smoke_passed',checks=checks,test_pixels_read=False,seconds=time.monotonic()-started,protocol_sha256=sha(a.protocol))
 with (out/'report.json').open('x') as f:f.write(json.dumps(report,indent=2))
 print(json.dumps(report),flush=True);raise SystemExit(0)

assert a.manifest and a.audit
m=json.loads(Path(a.manifest).read_text());audit=json.loads(Path(a.audit).read_text());test=m['records'];assert len(test)==256 and audit['status']=='fixed256_source_GT_noise_and_target_audit_passed' and audit['manifest_sha256']==sha(a.manifest)
assert len({r['source_id'] for r in test})==256
with (out/'EVALUATION_STARTED').open('x') as f:f.write('Fixed allweights, protocol and256pixel sources before first score; no automatic rerun.')
first8={r['source_id'] for r in test[:8]};recordfile=(out/'per_source.jsonl').open('x')
def record(row):rows.append(row);recordfile.write(json.dumps(row)+'\n');recordfile.flush()
for branch in ('reference','mean','measurement_best'):
 for name,k,dc in [('raw_HIO',0,False),('HIO_5DC',0,True),('timing_capped_extraHIO_5DC',protocol['HIO_controls']['extra_iterations'],True)]:
  warm=load(trainrows[0]);hio(warm,branch,k,dc)
  for meta in test:
   b=load(meta);base=dict(source_id=meta['source_id'],method=name,branch=branch,extra_HIO_iterations=k,dc_iterations=5 if dc else 0,initializer_seconds=meta['initializer_seconds'])
   if meta['common_initializer_failure']:record(dict(base,online_refinement_median_seconds=None,estimated_total_online_seconds=None,**failed('common_initializer_nonfinite')));continue
   x,sec,ts=timed(lambda:hio(b,branch,k,dc),3);q=quality(x,b) if torch.isfinite(x).all() else failed('nonfinite_prediction')
   record(dict(base,online_refinement_median_seconds=sec,online_times=ts,estimated_total_online_seconds=sec+meta['initializer_seconds'],**q))
   if not q['failed'] and name=='HIO_5DC' and branch=='mean':picture(meta['source_id'],'GT_raw_orientation',b['gt']);picture(meta['source_id'],'HIO_mean_5DC_raw_orientation',x)
  print(json.dumps(dict(completed_baseline=name,branch=branch,rows=len(rows),seconds=time.monotonic()-started)),flush=True)
for seed in (2027,2029):
 for mode in ('indi','fm'):
  for kind in ('best','last'):
   key=str(seed)+'_'+mode+'_'+kind;model=model_from(key);cm=proof['checkpoints'][key]
   for nfe in (1,4,8):
    for policy in ('per_stage5','fixed_total40'):
     warm=load(trainrows[0]);forward(model,warm,trainrows[0]['source_id'],nfe,policy);primary=nfe==4 and policy=='per_stage5'
     for meta in test:
      b=load(meta);base=dict(source_id=meta['source_id'],method=mode,seed=seed,kind=kind,nfe=nfe,dc_policy=policy,dc_iterations=5*nfe if policy=='per_stage5' else 40,checkpoint_step=cm['step'],checkpoint_sha256=cm['sha256'],initializer_seconds=meta['initializer_seconds'])
      if meta['common_initializer_failure']:record(dict(base,online_refinement_median_seconds=None,estimated_total_online_seconds=None,**failed('common_initializer_nonfinite')));continue
      x,sec,ts=timed(lambda:forward(model,b,meta['source_id'],nfe,policy),3 if primary else 1);q=quality(x,b) if torch.isfinite(x).all() else failed('nonfinite_prediction')
      record(dict(base,online_refinement_median_seconds=sec,online_times=ts,estimated_total_online_seconds=sec+meta['initializer_seconds'],**q))
      if kind=='best' and primary and not q['failed']:picture(meta['source_id'],key+'_4NFE20DC_raw_orientation',x)
     print(json.dumps(dict(completed_model=key,nfe=nfe,dc_policy=policy,rows=len(rows),seconds=time.monotonic()-started)),flush=True)
   del model;gc.collect();torch.cuda.empty_cache()
recordfile.close();assert len(rows)==256*(9+8*6)
groups={}
for row in rows:
 key=(row['method'],row.get('seed',''),row.get('kind',row.get('branch','')),row.get('nfe',''),row.get('dc_policy',''));groups.setdefault(key,[]).append(row)
for key,rr in groups.items():
 assert len(rr)==256 and len({r['source_id'] for r in rr})==256
 good=[r for r in rr if not r['failed']];times=[r['estimated_total_online_seconds'] for r in rr if r['estimated_total_online_seconds'] is not None]
 summaries.append(dict(method=key[0],seed=key[1],kind_or_branch=key[2],nfe=key[3],dc_policy=key[4],sources=256,failures=len(rr)-len(good),mean_flip_PSNR=statistics.mean(r['flip_psnr'] for r in rr),mean_flip_MSE=statistics.mean(r['flip_mse'] for r in rr),finite_only_flip_PSNR=statistics.mean(r['flip_psnr'] for r in good) if good else None,mean_raw_PSNR=statistics.mean(r['raw_psnr'] for r in rr),mean_orbit_PSNR=statistics.mean(r['aligned_psnr'] for r in rr),finite_only_amplitude_residual=statistics.mean(r['amp_residual'] for r in good) if good else None,median_estimated_online_seconds=statistics.median(times) if times else None))
paired=[]
for kind in ('best','last'):
 for nfe in (1,4,8):
  for policy in ('per_stage5','fixed_total40'):
   for seed in (2027,2029):
    methods={m:{r['source_id']:r for r in rows if r['method']==m and r.get('seed')==seed and r.get('kind')==kind and r.get('nfe')==nfe and r.get('dc_policy')==policy} for m in ('indi','fm')}
    assert set(methods['indi'])==set(methods['fm'])
    dd=[methods['fm'][sid]['flip_psnr']-methods['indi'][sid]['flip_psnr'] for sid in methods['fm']]
    paired.append(dict(kind=kind,nfe=nfe,dc_policy=policy,seed=seed,mean_FM_minus_InDI_dB=statistics.mean(dd),degraded_more_than0p01_count=sum(v<-.01 for v in dd),worst_delta=min(dd),per_source_deltas=dd,source_ids=list(methods['fm'])))
baseline_comparisons=[]
baselines={key:{r['source_id']:r for r in rr} for key,rr in groups.items() if key[0] not in ('indi','fm')}
for key,rr in groups.items():
 if key[0] not in ('indi','fm'):continue
 for baseline,lookup in baselines.items():
  dd=[r['flip_psnr']-lookup[r['source_id']]['flip_psnr'] for r in rr]
  baseline_comparisons.append(dict(method=key[0],seed=key[1],kind=key[2],nfe=key[3],dc_policy=key[4],baseline=baseline[0],baseline_branch=baseline[2],mean_delta_dB=statistics.mean(dd),degraded_more_than0p01_count=sum(v<-.01 for v in dd),worst_delta=min(dd)))
result=dict(status='independent_test_completed_all_frozen_variants',protocol_sha256=sha(a.protocol),test_manifest_sha256=sha(a.manifest),lock_proof_sha256=sha(a.lock),sources=256,source_ids=[r['source_id'] for r in test],common_initializer_failures=m['initializer_failures'],summaries=summaries,paired=paired,baseline_comparisons=baseline_comparisons,rows=rows,seconds=time.monotonic()-started,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,qualitative_source_ids=[r['source_id'] for r in test[:8]],qualitative_orientation='Raw prediction orientation, noGT alignment; source PNGs retained only on ownedGPU workspace',all_variants_reported=True,test_used_for_selection=False,limits=protocol['limitations'])
with (out/'results.json').open('x') as f:f.write(json.dumps(result,indent=2))
print(json.dumps(dict(status=result['status'],sources=256,rows=len(rows),seconds=result['seconds'],primary=[x for x in paired if x['kind']=='best' and x['nfe']==4 and x['dc_policy']=='per_stage5'])),flush=True)
