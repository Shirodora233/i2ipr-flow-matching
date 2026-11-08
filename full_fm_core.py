"""Full-budget conditional InDI/FM replacement. No prior analytic FM residual."""
import torch
from torch import nn
from upstream_adapter import (build_model, fft2d, ifft2d, zero_padding_twice,
                              crop_center_half, apply_image_constraint_hio)

class FullConditionalSolver(nn.Module):
 def __init__(self,mode,T=32,bridge_sigma=3/255,dc_steps=5,dc_precision="float32"):
  super().__init__();assert mode in ('indi','fm','matched_endpoint');w=build_model();self.raw=w.denoiser.model;self.mode=mode;self.T=T;self.bridge_sigma=bridge_sigma;self.dc_steps=dc_steps;assert dc_precision in ("float32","float64");self.dc_precision=dc_precision
  self.lam=nn.Parameter(torch.logspace(-.64,-.10,T,device='cpu').cuda())
 def field(self,x,hio,t):
  inp=torch.cat([x,hio],1);q=self.raw(inp,t,return_dict=False)[0].float();tau=t.float().view(-1,1,1,1)/self.T
  if self.mode=='indi':D=q+inp.float().mean(1,keepdim=True);return D,(D-x)/tau
  # Both FM and its optional endpoint-weighted control share exactly this field.
  v=q;return x+tau*v,v
 def dc(self,D,amp,support,t,steps=None):
  # Same original native255 arithmetic and endpoint-then-Euler ordering. No
  # smoothing, sigmoid lambda or stop-gradient hidden in the faithful control.
  steps=self.dc_steps if steps is None else steps
  if steps==0:return D
  dtype=torch.float64 if self.dc_precision=='float64' else torch.float32;z=zero_padding_twice(D.to(dtype)*255);b=amp.to(dtype)*255;lam=self.lam[t.long()-1].to(dtype).view(-1,1,1,1);bn=lam*b+(1-lam)*fft2d(z).abs()
  for _ in range(steps):z=apply_image_constraint_hio(ifft2d(torch.polar(bn,fft2d(z).angle())).real,z,support,beta=.9)
  return crop_center_half(z).clamp(0,255)/255
 def training_loss(self,batch,t,epsilon):
  y=batch['target'].float();h=batch['hio'].float();m=h.mean(1,keepdim=True);z0=m+self.bridge_sigma*epsilon;tau=t.float().view(-1,1,1,1)/self.T;x=tau*z0+(1-tau)*y
  D,v=self.field(x,h,t);post=self.dc(D,batch['amp'],batch['support'],t)
  if self.mode=='fm':primary=(v-(y-z0)).square().mean() # TRUE velocity MSE; no tau²
  else:primary=(D-y).square().mean()
  physical=(post-y).square().mean();total=.5*(primary+physical)*255**2
  return total,dict(primary=float(primary.detach()),post_dc=float(physical.detach()),endpoint_mse=float((D-y).square().mean().detach()),velocity_mse=float((v-(y-z0)).square().mean().detach()))
 @torch.no_grad()
 def infer(self,batch,nfe=4,noise=None,dc_policy='per_stage5'):
  assert nfe in (1,4,8) and self.T%nfe==0;h=batch['hio'];m=h.mean(1,keepdim=True);noise=torch.randn_like(m) if noise is None else noise;x=m+self.bridge_sigma*noise
  ks=[5]*nfe if dc_policy=='per_stage5' else [40//nfe]*nfe
  assert dc_policy in ('per_stage5','fixed_total40')
  for j in range(nfe):
   idx=self.T-j*(self.T//nfe);t=torch.full((h.shape[0],),idx,device=h.device,dtype=torch.long);D,v=self.field(x,h,t);projected=self.dc(D,batch['amp'],batch['support'],t,ks[j]);tau=idx/self.T
   # FM velocity plus endpoint physics correction; withoutDC exactly Euler v.
   effective=v+(projected-D)/tau if self.mode in ('fm','matched_endpoint') else (projected-x)/tau;x=(x+effective/nfe).to(h.dtype)
  return x

def times_and_noise(seed,global_update,micro_offset,shape,device,T=32):
 gen=torch.Generator(device='cpu').manual_seed(seed*1000000007+global_update*100003+micro_offset*97)
 n=shape[0];mask=torch.rand(n,generator=gen)<=2/3;t=torch.randint(1,T+1,(n,),generator=gen);t=torch.where(mask,torch.full_like(t,T),t);noise=torch.randn(shape,generator=gen);return t.to(device),noise.to(device)
