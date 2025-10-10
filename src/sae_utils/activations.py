import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import override


class TopKActivation(Module):
    def __init__(self, k: int) -> None:
        """Initialize the TopKActivation module.

        Args:
            k: The number of top activations to keep.

        """
        super().__init__()
        self.k = k

    def forward(self, z: Tensor) -> Tensor:
        """Apply a top-k selection to the input tensor along the last dimension.

        This method selects the top `k` largest values from the input tensor `z` along
        the last dimension, and returns a tensor of the same shape as `z` where only the
        top-k values are retained at their respective positions, and all other elements
        are set to zero.

        Args:
            z: Input tensor of arbitrary shape.

        Returns:
            Output tensor of the same shape as `z`, with only the top-k values retained
                along the last dimension and all other elements set to zero.

        Raises:
            ValueError: If `k` is greater than the size of the last dimension of `z`.

        """
        if self.k > z.shape[-1]:
            raise ValueError(
                f"k cannot be greater than the last dimension of the input tensor. "
                f"Got k={self.k} and input tensor with last dimension size "
                f"{z.shape[-1]}",
            )

        topk_values, topk_indices = torch.topk(z, self.k, dim=-1)
        output = torch.zeros_like(z).scatter_(
            dim=-1,
            index=topk_indices,
            src=topk_values,
        )
        return output

    def __str__(self) -> str:
        """Return a string representation of the TopKActivation module."""
        return f"{self.__class__.__name__}(k={self.k})"


class AbsTopKActivation(TopKActivation):
    @override
    def forward(self, z: Tensor) -> Tensor:
        """Apply an abs top-k selection to the input tensor along the last dimension.

        This method selects the top `k` largest absolute values from the input tensor
        `z` along the last dimension, and returns a tensor of the same shape as `z`
        where only the top-k absolute values are retained at their respective positions,
        and all other elements are set to zero. To be explicit, the original sign of the
        values is preserved.

        Args:
            z: Input tensor of arbitrary shape.

        Returns:
            Output tensor of the same shape as `z`, with only the top-k absolute values
                retained along the last dimension and all other elements set to zero.

        Raises:
            ValueError: If `k` is greater than the size of the last dimension of `z`.

        """
        if self.k > z.shape[-1]:
            raise ValueError(
                f"k cannot be greater than the last dimension of the input tensor. "
                f"Got k={self.k} and input tensor with last dimension size "
                f"{z.shape[-1]}",
            )
        neg_values_mask = z < 0
        unsigned_topk = super().forward(z.abs())
        signed_topk = torch.where(neg_values_mask, -unsigned_topk, unsigned_topk)
        return signed_topk


def update_dead_neuron_counts(z: Tensor, prev_counts: Tensor) -> Tensor:
    """Update the count of dead neurons based on the current batch activations.

    # ! Check the resets to zero part
    A neuron is considered "dead" in a batch if all its activations are zero.
    This function increments the dead neuron count for each neuron that is dead in the
    current batch, and resets the count to zero for neurons that are active.

    Args:
        z: The activation tensor of shape (batch_size, num_neurons).
        prev_counts: The tensor containing previous dead neuron counts for each
            neuron.

    Returns:
        Updated dead neuron counts for each neuron.

    """
    # ! Right now this function increments the count for dead neurons and keeps the
    # ! count for active neurons. Is this the desired behavior? Or should we reset the
    # ! count to zero when a neuron activates?
    dead_act = (z == 0).all(dim=0).to(dtype=torch.int)  # 1 when dead, 0 when active
    # ! WARNING: unused code
    # count = prev_counts * dead_act  # Reset count if neuron is active in this batch
    count = prev_counts + dead_act
    return count


if __name__ == "__main__":  # pragma: no cover[demonstration-code]
    z = torch.tensor(
        [  #  D    A    D    A
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.1],
        ],
    )
    prev_counts = torch.tensor([10, 20, 30, 40])
    dead_act = (z == 0).all(dim=0).to(dtype=torch.int)
    print(dead_act)
    print(prev_counts * dead_act)
    print(prev_counts + dead_act)
