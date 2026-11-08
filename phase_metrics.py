"""Fixed raw/orbit/measurement metrics; no truth-dependent reconstruction."""
import math
import numpy as np

def metrics(pred,truth,observed):
    def arr(x):
        if hasattr(x,'detach'):x=x.detach().cpu().numpy()
        return np.asarray(x,dtype=np.float64).squeeze()
    x,y,b=map(arr,(pred,truth,observed));assert x.ndim==2 and x.shape==y.shape
    h,w=x.shape;assert h%2==0 and w%2==0
    px=np.pad(x,((h//2,h//2),(w//2,w//2)))
    py=np.pad(y,((h//2,h//2),(w//2,w//2)))
    raw=float(np.mean((x-y)**2));best=float('inf');transform=None
    for flip,q in ((False,px),(True,px[::-1,::-1])):
        corr=np.fft.ifft2(np.fft.fft2(py)*np.conj(np.fft.fft2(q))).real
        dy,dx=np.unravel_index(np.argmax(corr),q.shape)
        mse=float(np.sum((np.roll(q,(int(dy),int(dx)),(0,1))-py)**2)/(h*w))
        if mse<best:best=mse;transform=dict(flip=flip,dy=int(dy),dx=int(dx))
    amp=np.abs(np.fft.fft2(px))/(2*np.sqrt(px.size))
    residual=float(np.linalg.norm(amp-b)/max(np.linalg.norm(b),1e-12))
    return dict(raw_mse=raw,raw_psnr=-10*math.log10(max(raw,1e-12)),
                aligned_mse=best,aligned_psnr=-10*math.log10(max(best,1e-12)),
                amp_residual=residual,transform=transform)
