import numpy as np

import pytest

from look.methods.participant_statistics import (participant_weights,fit_node_normalization,
 fit_weighted_pca,balanced_budget,feasibility_table,grouped_folds,aligned_group)

from look.methods.grouped_ridge import fit_ridge,select_conditional_ridge

def test_duplicate_eye_preserves_person_weighted_statistics_and_pca():
    x=np.array([[1.,4.,2.],[2.,7.,1.],[4.,2.,5.],[8.,9.,6.]])
    a=fit_node_normalization(x,np.ones(4))
    y=x[[0,1,1,2,3]];w=participant_weights(['a','b','b','c','d'])
    b=fit_node_normalization(y,w)
    np.testing.assert_allclose(a.mean,b.mean)
    np.testing.assert_allclose(a.coordinate_scale,b.coordinate_scale)
    assert np.average(np.sum(a.encode(x)**2,axis=1))==pytest.approx(1.)
    p=fit_weighted_pca(a.encode(x),np.ones(4),2,max_workspace_bytes=10**7)
    q=fit_weighted_pca(b.encode(y),w,2,max_workspace_bytes=10**7)
    np.testing.assert_allclose(p.components,q.components,atol=1e-12)
    residual=p.decode_residual(np.zeros((4,2)))
    np.testing.assert_array_equal(x+a.decode_residual(residual),x)

def test_rank_and_identity_are_never_silently_reduced():
    assert balanced_budget(5,[1,5])==(1,4)
    with pytest.raises(ValueError):balanced_budget(1,[3,3])
    rows=feasibility_table([[4,3],[3,3]],[7,6],{'workspace':8})
    assert next(r for r in rows if r['q']==4)['feasible']
    assert not next(r for r in rows if r['q']==8)['feasible']
    with pytest.raises(MemoryError):fit_weighted_pca(np.eye(4),np.ones(4),2,max_workspace_bytes=1)
    with pytest.raises(ValueError,match='rank'):fit_weighted_pca(np.ones((4,2)),np.ones(4),1,max_workspace_bytes=10**6)
    with pytest.raises(ValueError,match='identities'):aligned_group([np.ones((2,1))]*2,[['aL','aR'],['a','b']])

def test_group_folds_are_balanced_and_ridge_matches_augmented_least_squares():
    folds=grouped_folds(list(range(31)),[i%2 for i in range(31)],[1+i%2 for i in range(31)],seed=3416)
    assert max(list(folds.values()).count(i) for i in range(5))-min(list(folds.values()).count(i) for i in range(5))<=1
    rng=np.random.default_rng(12);x=rng.normal(size=(50,3));y=rng.normal(size=(50,2));w=rng.uniform(.3,2,50)
    model=fit_ridge(x,y,w,1.)
    xm=np.average(x,weights=w,axis=0);ym=np.average(y,weights=w,axis=0)
    design=np.vstack([(x-xm)*np.sqrt(w[:,None]/w.sum()),np.eye(3)*np.sqrt(model.absolute_ridge)])
    target=np.vstack([(y-ym)*np.sqrt(w[:,None]/w.sum()),np.zeros((3,2))])
    np.testing.assert_allclose(model.matrix,np.linalg.lstsq(design,target,rcond=None)[0],atol=1e-12)
    block=fit_ridge(x,y,w,1.,[((0,),(0,)),((1,2),(1,))])
    assert block.matrix[0,1]==0 and block.matrix[1,0]==0

def test_cv_refits_each_fold_and_rejects_globally_fitted_statistics():
    ids=range(10);folds={i:i%5 for i in ids};seen=[]
    def prepare(fit,held):
        seen.append((fit,held))
        def data(ix):
            x=np.array(sorted(ix),dtype=float)[:,None];return x,2*x+3,np.ones(len(x))
        return dict(fit=data(fit),held=data(held),audit={k:sorted(fit) for k in ('statistics','pca','upstream')})
    result=select_conditional_ridge(folds,prepare)
    assert len(seen)==5 and all(not a&b for a,b in seen)
    assert result['relative_ridge']==1e-8
    def leaked(fit,held):
        d=prepare(fit,held);d['audit']['pca']=list(ids);return d
    with pytest.raises(ValueError,match='not refitted'):select_conditional_ridge(folds,leaked)
    # Constant design is intercept-only; stronger ridge wins exact score ties.
    def constant(fit,held):
        d=prepare(fit,held)
        d['fit']=(np.zeros((len(fit),1)),np.ones((len(fit),1)),np.ones(len(fit)))
        d['held']=(np.zeros((len(held),1)),np.ones((len(held),1)),np.ones(len(held)))
        return d
    assert select_conditional_ridge(folds,constant)['relative_ridge']==100.

def test_each_node95_not_group95_and_not_equal_allocation():
    from look.methods.participant_statistics import select_node95_matched_rank
    args=dict(node_eigenvalues=[[.97,.02,.01],[.5,.3,.16,.04]],
              joint_eigenvalues=[1.,.5,.3,.16,.02,.01,.01],node_total_variances=[1.,1.],
              joint_total_variance=2.,fold_node_ranks=[[3,4]]*5,fold_joint_ranks=[7]*5,resource_max_rank=8)
    row=select_node95_matched_rank(**args)
    assert row['feasible'] and row['selected']['allocation']==(1,3) and row['selected']['q']==4
    np.testing.assert_allclose(row['selected']['per_node_retained'],[.97,.96],rtol=0,atol=2e-16)
    blocked=select_node95_matched_rank(**{**args,'resource_max_rank':3})
    assert not blocked['feasible'] and blocked['selected']['q']==4
    # Strong cross-node redundancy can prevent a full-rank common budget.
    redundant=select_node95_matched_rank(**{**args,'joint_eigenvalues':[1.5,.5,0.,0.], 'fold_joint_ranks':[2]*5})
    assert not redundant['feasible'] and redundant['selected']['allocation']==(1,3)
