import numpy as np
import pytest
from look.data.interface import AlignedFeatures


def args():
    return ({"a": np.ones((2, 3)), "b": np.zeros((2, 2, 2))},
            {"a": ["SYNTHETIC_1", "SYNTHETIC_2"], "b": ["SYNTHETIC_1", "SYNTHETIC_2"]},
            {"a": ["GROUP_1", "GROUP_1"], "b": ["GROUP_1", "GROUP_1"]})


def test_shapes_and_immutable_snapshot():
    a, k, g = args(); result = AlignedFeatures.from_modalities(a, k, g)
    assert result.flattened("b").shape == (2, 4)
    a["a"][:] = 9
    assert np.all(result.arrays["a"] == 1)
    with pytest.raises(ValueError): result.arrays["a"][0] = 3
    with pytest.raises(TypeError): result.arrays["a"] = np.zeros((2, 3))


def test_alignment_missingness_and_finite_validation():
    a, k, g = args(); k["b"].reverse()
    with pytest.raises(ValueError, match="order"): AlignedFeatures.from_modalities(a, k, g)
    a, k, g = args(); g["b"][0] = "OTHER"
    with pytest.raises(ValueError, match="identities"): AlignedFeatures.from_modalities(a, k, g)
    a, k, g = args(); a["a"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"): AlignedFeatures.from_modalities(a, k, g)
    a, k, g = args()
    with pytest.raises(ValueError, match="Boolean"):
        AlignedFeatures.from_modalities(a, k, g, {"a": np.ones(2), "b": np.zeros(2)})
