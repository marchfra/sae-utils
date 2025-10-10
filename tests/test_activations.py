import pytest
import torch
from torch.nn import Module

from sae_utils.activations import (
    AbsTopKActivation,
    TopKActivation,
    update_dead_neuron_counts,
)


def test_topkactivation_init_sets_k() -> None:
    k = 5
    activation = TopKActivation(k)
    assert activation.k == k


def test_topkactivation_init_inherits_module() -> None:
    activation = TopKActivation(3)
    assert isinstance(activation, Module)


def test_topkactivation_forward_returns_topk_values_and_zeros_elsewhere() -> None:
    activation = TopKActivation(2)
    z = torch.tensor(
        [
            [1.0, 3.0, 2.0, 2.0],
            [4.0, 0.0, 5.0, 2.0],
            [-1.0, -3.0, -2.0, -4.0],
        ],
    )
    output = activation.forward(z)
    expected = torch.tensor(
        [
            [0.0, 3.0, 2.0, 0.0],
            [4.0, 0.0, 5.0, 0.0],
            [-1.0, 0.0, -2.0, 0.0],
        ],
    )
    assert torch.allclose(output, expected)


def test_topkactivation_forward_with_k_equals_1() -> None:
    activation = TopKActivation(1)
    z = torch.tensor([[1.0, 3.0, 2.0]])
    output = activation.forward(z)
    expected = torch.tensor([[0.0, 3.0, 0.0]])
    assert torch.allclose(output, expected)


def test_topkactivation_forward_with_k_equals_tensor_last_dim() -> None:
    z = torch.tensor([[1.0, 3.0, 2.0]])
    activation = TopKActivation(z.shape[-1])
    output = activation.forward(z)
    assert torch.allclose(output, z)


def test_topkactivation_forward_with_k_greater_than_tensor_last_dim_raises() -> None:
    activation = TopKActivation(4)
    z = torch.tensor([[1.0, 3.0, 2.0]])
    with pytest.raises(
        ValueError,
        match="k cannot be greater than the last dimension of the input tensor",
    ):
        activation.forward(z)


def test_topkactivation_forward_with_tied_values() -> None:
    activation = TopKActivation(2)
    z = torch.tensor([[1.0, 3.0, 3.0, 2.0]])
    output = activation.forward(z)
    expected = torch.tensor(
        [[0.0, 3.0, 3.0, 0.0]],
    )  # torch.topk returns first occurrences in case of ties
    assert torch.allclose(output, expected)


def test_topkactivation_forward_with_negative_values() -> None:
    # torch.topk looks at largest values, not largest absolute values. This is
    # generally not a problem, since with reasonable choices of k, it's highly unlikely
    # that the top-k values are all negative
    activation = TopKActivation(2)
    z = torch.tensor([[-1.0, -3.0, -2.0]])
    output = activation.forward(z)
    expected = torch.tensor([[-1.0, 0.0, -2.0]])
    assert torch.allclose(output, expected)


def test_topkactivation_forward_with_batch_dimension() -> None:
    activation = TopKActivation(1)
    z = torch.tensor([[1.0, 2.0], [3.0, 0.0]])
    output = activation.forward(z)
    expected = torch.tensor([[0.0, 2.0], [3.0, 0.0]])
    assert torch.allclose(output, expected)


def test_topkactivation_forward_with_k_zero_returns_zeros() -> None:
    activation = TopKActivation(0)
    z = torch.tensor([[1.0, 2.0, 3.0]])
    output = activation.forward(z)
    expected = torch.zeros_like(z)
    assert torch.allclose(output, expected)


def test_topkactivation_forward_preserves_shape() -> None:
    activation = TopKActivation(2)
    z = torch.randn(4, 5, 6)
    output = activation.forward(z)
    assert output.shape == z.shape


def test_topkactivation_str_returns_correct_string() -> None:
    k = 3
    activation = TopKActivation(k)
    assert str(activation) == f"TopKActivation(k={k})"


def test_abstopkactivation_forward_returns_topk_values_and_zeros_elsewhere() -> None:
    activation = AbsTopKActivation(2)
    z = torch.tensor(
        [
            [1.0, 3.0, 2.0, 2.0],
            [4.0, 0.0, 5.0, 2.0],
            [-1.0, -3.0, -2.0, -4.0],
        ],
    )
    output = activation.forward(z)
    expected = torch.tensor(
        [
            [0.0, 3.0, 2.0, 0.0],
            [4.0, 0.0, 5.0, 0.0],
            [0.0, -3.0, 0.0, -4.0],
        ],
    )
    assert torch.allclose(output, expected)


@pytest.mark.parametrize(
    ("z", "expected"),
    [
        (torch.tensor([[1.0, 3.0, 2.0]]), torch.tensor([[0.0, 3.0, 0.0]])),
        (torch.tensor([[-1.0, -3.0, -2.0]]), torch.tensor([[0.0, -3.0, 0.0]])),
    ],
)
def test_abstopkactivation_forward_with_k_equals_1(
    z: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    activation = AbsTopKActivation(1)
    output = activation.forward(z)
    assert torch.allclose(output, expected)


def test_abstopkactivation_forward_with_mixed_signs() -> None:
    activation = AbsTopKActivation(3)
    z = torch.tensor([[1, 2, 3, 4, -10]])
    output = activation.forward(z)
    expected = torch.tensor([[0, 0, 3, 4, -10]])
    assert torch.allclose(output, expected)


def test_abstopkactivation_forward_with_k_equals_tensor_last_dim() -> None:
    z = torch.tensor([[1.0, 3.0, 2.0]])
    activation = AbsTopKActivation(z.shape[-1])
    output = activation.forward(z)
    assert torch.allclose(output, z)


def test_abstopkactivation_forward_with_k_greater_than_tensor_last_dim_raises() -> None:
    activation = AbsTopKActivation(4)
    z = torch.tensor([[1.0, 3.0, 2.0]])
    with pytest.raises(
        ValueError,
        match="k cannot be greater than the last dimension of the input tensor",
    ):
        activation.forward(z)


def test_abstopkactivation_forward_with_tied_values() -> None:
    activation = AbsTopKActivation(2)
    z = torch.tensor([[1.0, 3.0, 3.0, 2.0]])
    output = activation.forward(z)
    expected = torch.tensor(
        [[0.0, 3.0, 3.0, 0.0]],
    )  # torch.topk returns first occurrences in case of ties
    assert torch.allclose(output, expected)


def test_abstopkactivation_forward_with_negative_values() -> None:
    activation = AbsTopKActivation(2)
    z = torch.tensor([[-1.0, -3.0, -2.0]])
    output = activation.forward(z)
    expected = torch.tensor([[0.0, -3.0, -2.0]])
    assert torch.allclose(output, expected)


def test_abstopkactivation_forward_with_batch_dimension() -> None:
    activation = AbsTopKActivation(1)
    z = torch.tensor([[1.0, 2.0], [3.0, 0.0]])
    output = activation.forward(z)
    expected = torch.tensor([[0.0, 2.0], [3.0, 0.0]])
    assert torch.allclose(output, expected)


def test_abstopkactivation_forward_with_k_zero_returns_zeros() -> None:
    activation = AbsTopKActivation(0)
    z = torch.tensor([[1.0, 2.0, 3.0]])
    output = activation.forward(z)
    expected = torch.zeros_like(z)
    assert torch.allclose(output, expected)


def test_abstopkactivation_forward_preserves_shape() -> None:
    activation = AbsTopKActivation(2)
    z = torch.randn(4, 5, 6)
    output = activation.forward(z)
    assert output.shape == z.shape


def test_abstopkactivation_str_returns_correct_string() -> None:
    k = 3
    activation = AbsTopKActivation(k)
    assert str(activation) == f"AbsTopKActivation(k={k})"


def test_update_dead_neuron_counts_increments_for_dead_neurons() -> None:
    z = torch.tensor([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
    prev_counts = torch.tensor([2, 5, 1])
    # Neuron 0 and 2 are dead (all zeros in batch), neuron 1 is active
    updated = update_dead_neuron_counts(z, prev_counts)
    expected = torch.tensor([3, 5, 2])
    assert torch.equal(updated, expected)


def test_update_dead_neuron_counts_resets_for_active_neurons() -> None:
    z = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    prev_counts = torch.tensor([4, 7])
    # Neuron 0 is dead, neuron 1 is active
    updated = update_dead_neuron_counts(z, prev_counts)
    expected = torch.tensor([5, 7])
    assert torch.equal(updated, expected)


def test_update_dead_neuron_counts_all_neurons_dead() -> None:
    z = torch.zeros(3, 4)
    prev_counts = torch.tensor([1, 2, 3, 4])
    updated = update_dead_neuron_counts(z, prev_counts)
    expected = torch.tensor([2, 3, 4, 5])
    assert torch.equal(updated, expected)


def test_update_dead_neuron_counts_no_neurons_dead() -> None:
    z = torch.ones(2, 3)
    prev_counts = torch.tensor([0, 1, 2])
    updated = update_dead_neuron_counts(z, prev_counts)
    expected = torch.tensor([0, 1, 2])
    assert torch.equal(updated, expected)


def test_update_dead_neuron_counts_mixed_batch() -> None:
    z = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    prev_counts = torch.tensor([1, 2, 3])
    # Neuron 0 and 1 are dead, neuron 2 is active
    updated = update_dead_neuron_counts(z, prev_counts)
    expected = torch.tensor([2, 3, 3])
    assert torch.equal(updated, expected)


def test_update_dead_neuron_counts_with_int_tensor() -> None:
    z = torch.tensor([[0, 0], [0, 1]], dtype=torch.int)  # 0 is dead, 1 is active
    prev_counts = torch.tensor([10, 20])
    updated = update_dead_neuron_counts(z, prev_counts)
    expected = torch.tensor([11, 20])
    assert torch.equal(updated, expected)
