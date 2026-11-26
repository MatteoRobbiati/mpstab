"""Pauli-string products of :mod:`mpstab.pauli`, against stim as a reference."""

import numpy as np
import pytest
import stim

from mpstab.pauli import multiply_codes, pauli_codes, pauli_labels, pauli_product


def _stim_product(*paulis):
    product = stim.PauliString(paulis[0])
    for pauli in paulis[1:]:
        product = product * stim.PauliString(pauli)
    return str(product).lstrip("+-i").replace("_", "I"), complex(product.sign)


@pytest.mark.parametrize("seed", range(5))
def test_products_match_stim(seed):
    rng = np.random.default_rng(seed)
    for _ in range(60):
        n, k = int(rng.integers(1, 7)), int(rng.integers(1, 5))
        paulis = ["".join(rng.choice(list("IXYZ"), n)) for _ in range(k)]
        assert pauli_product(*paulis) == _stim_product(*paulis)


def test_multiply_codes_broadcasts():
    left = pauli_codes(["XY", "ZZ", "II"])
    codes, phases = multiply_codes(left, pauli_codes("YX"))
    assert pauli_labels(codes) == ["ZZ", "XY", "YX"]
    expected = [_stim_product(p, "YX")[1] for p in ("XY", "ZZ", "II")]
    np.testing.assert_allclose(1j**phases, expected)


@pytest.mark.parametrize("bad", ["x", "XAZ", "X Z", "+X"])
def test_invalid_characters_raise(bad):
    with pytest.raises(ValueError, match="Invalid characters"):
        pauli_codes(bad)
    with pytest.raises(ValueError, match="Invalid characters"):
        pauli_product("I" * len(bad), bad)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="different lengths"):
        pauli_product("XI", "YZZ")
    with pytest.raises(ValueError, match="different lengths"):
        pauli_codes(["XI", "YZZ"])


def test_no_strings():
    assert pauli_codes([]).shape == (0, 0)
    assert pauli_labels(pauli_codes([])) == []
    with pytest.raises(ValueError, match="at least one"):
        pauli_product()
