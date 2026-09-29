"""Weighted affine residual fitting and conditional participant-CV selection."""
from dataclasses import dataclass
import numpy as np
from look.methods.participant_statistics import weighted_center

RELATIVE_RIDGES = tuple(10. ** i for i in range(-8, 3))


@dataclass(frozen=True)
class RidgeMap:
    x_mean: np.ndarray
    intercept: np.ndarray
    matrix: np.ndarray
    relative_ridge: float
    absolute_ridge: float
    effective_df: float

    def predict(self, x):
        return (x - self.x_mean) @ self.matrix + self.intercept


def fit_ridge(x, target, weights, relative_ridge, blocks=None):
    xm, xc, w = weighted_center(x, weights)
    ym, yc, wy = weighted_center(target, weights)
    if len(x) != len(target) or not np.array_equal(w, wy):
        raise ValueError('Residual fit row alignment differs')
    if relative_ridge not in RELATIVE_RIDGES:
        raise ValueError('Unregistered relative ridge')
    gram = xc.T @ (w[:, None] * xc)
    cross = xc.T @ (w[:, None] * yc)
    scale = float(np.trace(gram) / len(gram)) if len(gram) else 0.
    absolute = relative_ridge * scale
    matrix = np.zeros((x.shape[1], target.shape[1]), dtype=np.float64); df = float(target.shape[1])
    blocks = blocks if blocks is not None else [(tuple(range(x.shape[1])), tuple(range(target.shape[1])))]
    written = set()
    for ins, outs in blocks:
        ins, outs = tuple(ins), tuple(outs)
        if set(outs) & written or len(set(ins)) != len(ins) or len(set(outs)) != len(outs):
            raise ValueError('Overlapping/duplicate block specification')
        written.update(outs)
        if any(i < 0 or i >= x.shape[1] for i in ins) or any(i < 0 or i >= target.shape[1] for i in outs):
            raise ValueError('Block coordinate out of range')
        if not ins or not outs or scale == 0:
            continue
        g = gram[np.ix_(ins,ins)]
        matrix[np.ix_(ins,outs)] = np.linalg.solve(g + absolute*np.eye(len(ins)), cross[np.ix_(ins,outs)])
        df += len(outs) * float(np.trace(np.linalg.solve(g + absolute*np.eye(len(ins)), g)))
    return RidgeMap(xm, ym, matrix, relative_ridge, absolute, df)


def select_conditional_ridge(folds, prepare_fold):
    """prepare_fold(fit_ids, held_ids) refits stats/PCA/upstream on fit_ids only.

    Returns fit x/r/w, held x/r/w and an audit with actual fitted participant IDs.
    The callback is mandatory: globally prefit matrices are not accepted here.
    Errors are accumulated per participant mass, not averaged per fold/eye.
    """
    labels = sorted(set(folds.values()))
    if labels != list(range(5)):
        raise ValueError('Exactly five nonempty participant folds required')
    errors = {a: 0. for a in RELATIVE_RIDGES}; mass = 0.; audits = []
    for f in labels:
        fit_ids = frozenset(p for p,k in folds.items() if k != f)
        held_ids = frozenset(p for p,k in folds.items() if k == f)
        data = prepare_fold(fit_ids, held_ids)
        audit = data['audit']
        for component in ('statistics','pca','upstream'):
            if frozenset(audit[component]) != fit_ids:
                raise ValueError(f'Fold {f} {component} was not refitted on fit participants')
        x, r, w = data['fit']; hx, hr, hw = data['held']
        if not np.isfinite(hr).all() or not np.isfinite(hw).all() or np.any(hw <= 0):
            raise ValueError('Invalid held-out target/weights')
        for a in RELATIVE_RIDGES:
            model = fit_ridge(x, r, w, a, data.get('blocks'))
            error = np.sum((model.predict(hx)-hr)**2, axis=1)
            errors[a] += float(hw @ error)
        mass += float(np.sum(hw)); audits.append(audit)
    scores = {a: e/mass for a,e in errors.items()}
    if not all(np.isfinite(s) for s in scores.values()):
        raise ValueError('Nonfinite CV score')
    best = min(scores, key=lambda a:(scores[a], -a))
    return {'relative_ridge': best, 'scores': scores, 'fold_audits': audits,
            'interpretation': 'conditional internal selection; frozen host/path/upstream hyperparameters'}


def fit_moment_ridge(moments, relative_ridge, blocks=None):
    """Same affine objective as fit_ridge, including conditional noise moments."""
    if relative_ridge not in RELATIVE_RIDGES:
        raise ValueError('Unregistered relative ridge')
    stats=moments.centered(); gram=stats['xx']; cross=stats['xy']
    dx,dy=cross.shape; scale=float(np.trace(gram)/dx) if dx else 0.
    if not all(np.isfinite(v).all() for v in stats.values()) or scale<0:
        raise ValueError('Invalid centered paired moments')
    absolute=relative_ridge*scale; matrix=np.zeros((dx,dy)); df=float(dy)
    blocks=blocks if blocks is not None else [(tuple(range(dx)),tuple(range(dy)))]
    written=set()
    for ins,outs in blocks:
        ins,outs=tuple(ins),tuple(outs)
        if (set(outs)&written or len(set(ins))!=len(ins) or len(set(outs))!=len(outs)
            or any(i<0 or i>=dx for i in ins) or any(i<0 or i>=dy for i in outs)):
            raise ValueError('Invalid regression blocks')
        written.update(outs)
        if not ins or not outs or scale==0:continue
        g=gram[np.ix_(ins,ins)]; system=g+absolute*np.eye(len(ins))
        matrix[np.ix_(ins,outs)]=np.linalg.solve(system,cross[np.ix_(ins,outs)])
        df+=len(outs)*float(np.trace(np.linalg.solve(system,g)))
    return RidgeMap(stats['x_mean'],stats['y_mean'],matrix,relative_ridge,absolute,df)


def score_moment_ridge(model, moments):
    """Expected per-person projected residual SSE, including shared randomness."""
    s=moments.centered(); m=model.matrix
    bias=(s['x_mean']-model.x_mean)@m+model.intercept-s['y_mean']
    terms=(float(np.trace(m.T@s['xx']@m)), -2*float(np.sum(m*s['xy'])),
           float(np.trace(s['yy'])),float(bias@bias))
    score=sum(terms)
    tolerance=64*np.finfo(float).eps*max(1.,sum(abs(x) for x in terms))*max(1,len(m))
    if score < -tolerance:raise ValueError('Negative expected squared error beyond roundoff')
    return max(score,0.)
