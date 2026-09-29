"""All data in this example are generated; nothing is downloaded or saved."""
import numpy as np
from look.data.interface import AlignedFeatures
from look.methods.participant_statistics import fit_weighted_pca
from look.methods.grouped_ridge import fit_ridge

rng = np.random.default_rng(7)
keys = tuple(f"SYNTHETIC_{i:03d}" for i in range(40))
x = rng.normal(size=(40, 6))
y = 0.2 * x + 0.1
batch = AlignedFeatures.from_modalities(
    {"view_a": x, "view_b": y.reshape(40, 2, 3)},
    {"view_a": keys, "view_b": keys}, {"view_a": keys, "view_b": keys})
pca = fit_weighted_pca(batch.flattened("view_a"), np.ones(40), 3,
                       max_workspace_bytes=10_000_000)
z = pca.encode(x)
target = pca.residual(x, y)
mapping = fit_ridge(z, target, np.ones(40), 1e-4)
corrected = x + pca.decode_residual(mapping.predict(z))
assert corrected.shape == x.shape and np.isfinite(corrected).all()
print("Synthetic PCA and residual mapping completed.")
