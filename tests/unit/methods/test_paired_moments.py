import itertools
import numpy as np
from look.methods.paired_moments import equal_embracement,paired_residual,WeightedPairedMoments,expected_squared_error


def test_exact_shared_draw_enumeration_survives_affine_correction_and_logits():
    a=np.array([[1.,3.],[2.,5.]]);b=np.array([[4.,2.],[1.,8.]])
    full=equal_embracement(a,b,source_id="host/shared_draw");missing=equal_embracement(a+.5,b-1.,source_id="host/shared_draw")
    correction=np.array([[1.2,.3],[-.4,.8]]);head=np.array([[.5,-1.],[1.,2.]])
    transformed=missing.transform(correction,np.array([.2,-.1])).transform(head,np.array([1.,0.]))
    target=full.transform(head,np.array([1.,0.]))
    residual=paired_residual(target,transformed)
    m=WeightedPairedMoments(2,2);w=np.array([1.,.5]);m.update(transformed,residual,w)
    x=[];r=[];weights=[]
    for row in range(2):
        for eps in itertools.product((-1.,1.),repeat=2):
            eps=np.array(eps)
            xm=transformed.mean[row]+transformed.factor[row]@eps
            ym=target.mean[row]+target.factor[row]@eps
            x.append(xm);r.append(ym-xm);weights.append(w[row]/4)
    x=np.array(x);r=np.array(r);weights=np.array(weights);xm=np.average(x,weights=weights,axis=0);rm=np.average(r,weights=weights,axis=0)
    actual=m.centered()
    np.testing.assert_allclose(actual['xx'],(x-xm).T@(weights[:,None]*(x-xm))/weights.sum(),atol=1e-14)
    np.testing.assert_allclose(actual['xy'],(x-xm).T@(weights[:,None]*(r-rm))/weights.sum(),atol=1e-14)
    np.testing.assert_allclose(expected_squared_error(transformed,target),np.sum(r*r,axis=1).reshape(2,4).mean(1),atol=1e-14)
    # Mean-only fitting loses within-person stochastic variance.
    assert not np.allclose(actual['xx'],np.cov(transformed.mean.T,bias=True))
