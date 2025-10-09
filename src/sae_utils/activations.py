import torch
from torch import Tensor
from torch.nn import Module


class TopKActivation(Module):
    def __init__(self, k: int) -> None:
        """Initialize the TopKActivation module.

        Args:
            k (int): The number of top activations to keep.

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
            z (Tensor): Input tensor of arbitrary shape.

        Returns:
            Tensor: Output tensor of the same shape as `z`, with only the top-k
                values retained along the last dimension and all other elements set to
                zero.

        """
        topk_values, topk_indices = torch.topk(z, self.k, dim=-1)
        output = torch.zeros_like(z).scatter_(
            dim=-1,
            index=topk_indices,
            src=topk_values,
        )
        return output

    def __str__(self) -> str:
        """Return a string representation of the TopKActivation module."""
        return f"TopKActivation(k={self.k})"


def update_dead_neuron_counts(z: Tensor, prev_counts: Tensor) -> Tensor:
    """Update the count of dead neurons based on the current batch activations.

    A neuron is considered "dead" in a batch if all its activations are zero.
    This function increments the dead neuron count for each neuron that is dead in the
    current batch, and resets the count to zero for neurons that are active.

    Args:
        z (Tensor): The activation tensor of shape (batch_size, num_neurons).
        prev_counts (Tensor): The tensor containing previous dead neuron counts for each
            neuron.

    Returns:
        Tensor: Updated dead neuron counts for each neuron.

    """
    dead_act = (z == 0).all(dim=0).to(dtype=torch.int)
    count = prev_counts * dead_act  # Reset count if neuron is active in this batch
    count = prev_counts + dead_act
    return count
