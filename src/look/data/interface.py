"""In-memory contracts for source-specific adapters maintained by the caller."""
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
import numpy as np


@dataclass(frozen=True)
class AlignedFeatures:
    arrays: Mapping[str, np.ndarray]
    sample_keys: tuple[str, ...]
    group_keys: tuple[str, ...]
    missing: Mapping[str, np.ndarray]

    @classmethod
    def from_modalities(cls, arrays: Mapping[str, np.ndarray],
                        sample_keys: Mapping[str, Sequence[str]],
                        group_keys: Mapping[str, Sequence[str]],
                        missing: Mapping[str, np.ndarray] | None = None):
        if not arrays or set(arrays) != set(sample_keys) or set(arrays) != set(group_keys):
            raise ValueError("All modalities require explicit sample and group keys")
        names = tuple(arrays)
        keys = tuple(sample_keys[names[0]])
        groups = tuple(group_keys[names[0]])
        if not keys or any(not isinstance(k, str) or not k for k in (*keys, *groups)):
            raise ValueError("Keys must be nonempty opaque strings")
        if len(set(keys)) != len(keys) or len(groups) != len(keys):
            raise ValueError("Duplicate sample keys or inconsistent group rows")
        if missing is not None and set(missing) != set(names):
            raise ValueError("Missingness must be provided for every modality")
        values, masks = {}, {}
        for name in names:
            if tuple(sample_keys[name]) != keys or tuple(group_keys[name]) != groups:
                raise ValueError("Sample order or group identities differ across modalities")
            a = np.asarray(arrays[name])
            if a.ndim < 2 or a.shape[0] != len(keys) or any(s == 0 for s in a.shape[1:]):
                raise ValueError("Expected nonempty [sample, ...feature] arrays")
            if not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all():
                raise ValueError("Feature arrays must be finite numeric values")
            m = np.zeros(len(keys), dtype=bool) if missing is None else np.asarray(missing[name])
            if m.dtype != np.bool_ or m.shape != (len(keys),):
                raise ValueError("Missingness must be a Boolean vector per modality")
            # Own immutable snapshots, so caller mutation cannot break alignment.
            values[name] = a.copy(); values[name].flags.writeable = False
            masks[name] = m.copy(); masks[name].flags.writeable = False
        from types import MappingProxyType
        return cls(MappingProxyType(values), keys, groups, MappingProxyType(masks))

    def flattened(self, modality):
        return self.arrays[modality].reshape(len(self.sample_keys), -1)
