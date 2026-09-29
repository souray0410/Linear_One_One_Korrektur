"""Participant-weighted representation and rank contracts for LOOK v2.

The exact reference SVD is resource bounded. A full-cohort streaming backend
must prove equivalence before replacing it; this function never silently lowers q.
"""
from dataclasses import dataclass
from collections import Counter
import hashlib
import numpy as np

DIMENSIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def participant_weights(ids):
    counts = Counter(ids)
    if not counts or any(c not in (1, 2) for c in counts.values()):
        raise ValueError('Expected one participant row or one/two valid eye rows')
    return np.asarray([1. / counts[i] for i in ids], dtype=np.float64)


def aligned_group(arrays, row_keys):
    if not arrays or len(arrays) != len(row_keys):
        raise ValueError('Missing group identity')
    keys = tuple(row_keys[0])
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate participant/eye state key')
    if any(tuple(k) != keys for k in row_keys):
        raise ValueError('State row identities differ: eye/participant mixing forbidden')
    if any(len(a) != len(keys) for a in arrays):
        raise ValueError('State rows and identities differ')
    return np.concatenate([np.asarray(a).reshape(len(keys), -1) for a in arrays], axis=1)


def weighted_center(x, weights):
    x = np.asarray(x, dtype=np.float64); w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or w.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('Invalid finite positive-weight matrix')
    mean = np.average(x, weights=w, axis=0)
    return mean, x - mean, w / w.sum()


@dataclass(frozen=True)
class NodeNormalization:
    mean: np.ndarray
    coordinate_scale: np.ndarray
    energy: float
    active: np.ndarray

    def encode(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape[-1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError('Invalid feature coordinates')
        return (x - self.mean) / self.coordinate_scale / self.energy

    def decode_residual(self, delta):
        return delta * self.coordinate_scale * self.energy


def fit_node_normalization(x, weights):
    mean, centered, w = weighted_center(x, weights)
    variance = w @ centered ** 2
    active = variance > 0
    # Constant coordinates stay explicit and contribute zero to complete PCA.
    scale = np.where(active, np.sqrt(variance), 1.)
    energy = float(np.sqrt(np.sum(variance / scale ** 2)))
    if energy == 0:
        raise ValueError('Zero total node variance: register intercept-only degeneracy')
    if not np.isfinite(energy):
        raise ValueError('Nonfinite node energy')
    return NodeNormalization(mean, scale, energy, active)


@dataclass(frozen=True)
class WeightedPCA:
    mean: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    rank: int

    def encode(self, x):
        return (x - self.mean) @ self.components.T

    def residual(self, x, y):
        return (y - x) @ self.components.T

    def decode_residual(self, r):
        return r @ self.components


def fit_weighted_pca(x, weights, q, *, max_workspace_bytes):
    mean, centered, w = weighted_center(x, weights)
    n, d = centered.shape
    # Conservative allowance for dense reference matrices/SVD temporaries.
    required = 8 * (5*n*d + 4*min(n,d)**2 + 3*(n+d)*min(n,d))
    if required > max_workspace_bytes:
        raise MemoryError(f'Exact reference SVD requires bound {required}; budget {max_workspace_bytes}')
    _, s, vt = np.linalg.svd(centered * np.sqrt(w[:, None]), full_matrices=False)
    rank = int(np.count_nonzero(s > np.finfo(np.float64).eps * max(n, d) * s[0])) if len(s) and s[0] else 0
    if type(q) is not int or q < 1 or q > rank:
        raise ValueError(f'Requested q={q} exceeds centered numerical rank {rank}')
    components = vt[:q].copy()
    # Sign identity fixed; repeated-eigenvalue subspaces still need SHA binding.
    for row in components:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    return WeightedPCA(mean, components, s[:q], rank)


def balanced_budget(q, capacities):
    capacities = tuple(map(int, capacities))
    if not capacities or any(c < 1 for c in capacities) or q < len(capacities) or q > sum(capacities):
        raise ValueError('Cannot allocate at least one component per active node')
    sizes = [1] * len(capacities)
    while sum(sizes) < q:
        i = min((i for i in range(len(sizes)) if sizes[i] < capacities[i]), key=lambda i:(sizes[i], i))
        sizes[i] += 1
    return tuple(sizes)


def feasibility_table(fold_node_ranks, fold_joint_ranks, resource_limits):
    """Ranks include full training and every CV fit split; no dev scores allowed."""
    if not fold_node_ranks or len(fold_node_ranks) != len(fold_joint_ranks):
        raise ValueError('Missing fold ranks')
    widths = {len(row) for row in fold_node_ranks}
    if len(widths) != 1:
        raise ValueError('Fold node ordering differs')
    caps = tuple(min(row[i] for row in fold_node_ranks) for i in range(next(iter(widths))))
    rows = []
    for q in DIMENSIONS:
        reasons = []
        try:
            allocation = balanced_budget(q, caps)
        except ValueError as exc:
            allocation = None; reasons.append(str(exc))
        if q > min(fold_joint_ranks): reasons.append('joint PCA fold rank')
        for label, limit in sorted(resource_limits.items()):
            if q > limit: reasons.append('resource: ' + label)
        rows.append(dict(q=q, allocation=allocation, feasible=not reasons, excluded_reasons=reasons))
    return rows


def grouped_folds(participant_ids, labels, valid_eyes, *, seed, n_splits=5):
    """Deterministic stratified round-robin on one metadata row per participant."""
    ids = tuple(participant_ids)
    if len(ids) != len(set(ids)) or len(labels) != len(ids) or len(valid_eyes) != len(ids) or len(ids) < n_splits:
        raise ValueError('Invalid participant fold manifest')
    if any(e not in (1,2) for e in valid_eyes):
        raise ValueError('Invalid valid-eye count')
    strata = {}
    for p,y,e in zip(ids, labels, valid_eyes): strata.setdefault((int(y),int(e)), []).append(p)
    assignment = {}; offset = 0
    for stratum in sorted(strata):
        ordered = sorted(strata[stratum], key=lambda p: hashlib.sha256(f'{seed}:{p}'.encode()).digest())
        for i,p in enumerate(ordered): assignment[p] = (offset+i) % n_splits
        offset = (offset+len(ordered)) % n_splits
    return assignment


def select_node95_matched_rank(node_eigenvalues, joint_eigenvalues, node_total_variances,
                               joint_total_variance, *, fold_node_ranks, fold_joint_ranks,
                               resource_max_rank):
    """Approved rule: each node's smallest >=95% rank; joint uses their SUM.

    q_j is locked from complete full-training statistics and remains fixed in CV.
    It is not rebalanced and never replaced by joint PCA's own smaller 95% rank.
    Resource/fold conflicts are explicit failed admission, not dimension changes.
    """
    spectra=[np.asarray(s,dtype=np.float64) for s in node_eigenvalues]
    joint=np.asarray(joint_eigenvalues,dtype=np.float64);totals=np.asarray(node_total_variances,dtype=np.float64)
    if not spectra or totals.shape!=(len(spectra),) or np.any(totals<=0) or not np.isfinite(totals).all():
        raise ValueError('Zero/nonfinite total variance requires explicit degeneracy registration')
    if not np.isfinite(joint_total_variance) or joint_total_variance<=0 or not np.isclose(totals.sum(),joint_total_variance,rtol=1e-10,atol=0):
        raise ValueError('Measured node/joint covariance traces differ')
    for s,total in [*zip(spectra,totals),(joint,joint_total_variance)]:
        if s.ndim!=1 or not len(s) or not np.isfinite(s).all() or np.any(s<0) or np.any(np.diff(s)>0) or s.sum()>total*(1+1e-12):
            raise ValueError('Invalid measured spectrum')
    if len(fold_node_ranks)!=5 or len(fold_joint_ranks)!=5 or any(len(r)!=len(spectra) for r in fold_node_ranks):
        raise ValueError('All five fold rank records required')
    dims=[];retained=[];reasons=[]
    for j,(s,total) in enumerate(zip(spectra,totals)):
        hits=np.flatnonzero(np.cumsum(s)/total>=.95)
        if not len(hits):reasons.append(f'node {j}: measured spectrum has not reached95');continue
        k=int(hits[0])+1;dims.append(k);retained.append(float(s[:k].sum()/total))
    if len(dims)!=len(spectra):
        return dict(rule='per_node_energy95_joint_sum_v1',selected=None,feasible=False,reasons=reasons,dev_used=False)
    q=sum(dims);joint_retained=None
    if q>len(joint):reasons.append('insufficient measured joint spectrum; do not substitute its smaller95 rank')
    else:
        joint_retained=float(joint[:q].sum()/joint_total_variance)
        if joint_retained<.95:reasons.append('joint covariance inconsistent with >=95 node projection at same budget')
        if np.count_nonzero(joint>0)<q:reasons.append('joint effective rank smaller than matched total budget')
    for f,(nr,jr) in enumerate(zip(fold_node_ranks,fold_joint_ranks)):
        if any(k>r for k,r in zip(dims,nr)):reasons.append(f'fold {f} node rank')
        if q>jr:reasons.append(f'fold {f} joint rank')
    if q>resource_max_rank:reasons.append('resource maximum rank; never clamp selected dimensions')
    return dict(rule='per_node_energy95_joint_sum_v1',selected=dict(q=q,allocation=tuple(dims),
                per_node_retained=retained,joint_retained=joint_retained),feasible=not reasons,reasons=reasons,
                dev_used=False,fold_interpretation='full-train node dimensions fixed; fold-local statistics/PCA/upstream refitted')
