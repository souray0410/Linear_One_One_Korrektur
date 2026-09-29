"""Exact paired second moments through stochastic fusion and affine corrections.

X=mx+Ux epsilon, Y=my+Uy epsilon, E epsilon=0, Cov epsilon=I.
The same epsilon is essential: independent full/missing draws change the target.
Factors can be accumulated batchwise; no cohort-sized N x D x D cache required.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AffineRandomState:
    mean: np.ndarray
    factor: np.ndarray
    source_id: str

    def __post_init__(self):
        if not self.source_id:raise ValueError('Explicit shared random source identity required')
        if self.mean.ndim!=2 or self.factor.ndim!=3 or self.factor.shape[:2]!=self.mean.shape:
            raise ValueError('Conditional moment shape mismatch')
        if not np.isfinite(self.mean).all() or not np.isfinite(self.factor).all():
            raise ValueError('Nonfinite conditional moments')

    def transform(self, matrix, intercept):
        return AffineRandomState(self.mean@matrix+intercept,np.einsum('ndk,df->nfk',self.factor,matrix),self.source_id)


def equal_embracement(first,second,*,source_id):
    first,second=np.asarray(first,dtype=np.float64),np.asarray(second,dtype=np.float64)
    if first.shape!=second.shape or first.ndim!=2:raise ValueError('Docking sizes differ')
    gap=(first-second)/2
    return AffineRandomState((first+second)/2,gap[:,:,None]*np.eye(gap.shape[1])[None,:,:],source_id)


def paired_residual(full,missing):
    if full.source_id!=missing.source_id or full.mean.shape!=missing.mean.shape or full.factor.shape!=missing.factor.shape:
        raise ValueError('Paired draw coordinate identity differs')
    return AffineRandomState(full.mean-missing.mean,full.factor-missing.factor,full.source_id)


class WeightedPairedMoments:
    def __init__(self,dx,dy):
        self.mass=0.;self.sx=np.zeros(dx);self.sy=np.zeros(dy)
        self.xx=np.zeros((dx,dx));self.xy=np.zeros((dx,dy));self.yy=np.zeros((dy,dy))

    def update(self,x,y,weights):
        w=np.asarray(weights,dtype=np.float64)
        if x.source_id!=y.source_id or x.factor.shape[2]!=y.factor.shape[2] or len(x.mean)!=len(y.mean) or w.shape!=(len(x.mean),):
            raise ValueError('Pair/weight identities differ')
        if not np.isfinite(w).all() or np.any(w<=0):raise ValueError('Invalid participant weights')
        self.mass+=w.sum();self.sx+=w@x.mean;self.sy+=w@y.mean
        self.xx+=x.mean.T@(w[:,None]*x.mean)+np.einsum('n,ndk,nek->de',w,x.factor,x.factor)
        self.xy+=x.mean.T@(w[:,None]*y.mean)+np.einsum('n,ndk,nek->de',w,x.factor,y.factor)
        self.yy+=y.mean.T@(w[:,None]*y.mean)+np.einsum('n,ndk,nek->de',w,y.factor,y.factor)

    def centered(self):
        if self.mass<=0:raise ValueError('Empty moments')
        x,y=self.sx/self.mass,self.sy/self.mass
        return dict(x_mean=x,y_mean=y,xx=self.xx/self.mass-np.outer(x,x),
                    xy=self.xy/self.mass-np.outer(x,y),yy=self.yy/self.mass-np.outer(y,y))


def expected_squared_error(predicted,target):
    if predicted.source_id!=target.source_id or predicted.factor.shape!=target.factor.shape or predicted.mean.shape!=target.mean.shape:
        raise ValueError('Prediction/target source identities differ')
    return ((predicted.mean-target.mean)**2).sum(1)+((predicted.factor-target.factor)**2).sum((1,2))
