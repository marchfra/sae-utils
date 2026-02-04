import torch
from torch import Tensor
from torch.nn import Module


class TopK(Module):
    def __init__(self, k: int) -> None:
        """Initialize the TopK module.

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
            msg = (
                f"k cannot be greater than the last dimension of the input tensor. "
                f"Got k={self.k} and input tensor with last dimension size "
                f"{z.shape[-1]}"
            )
            raise ValueError(msg)

        topk_values, topk_indices = torch.topk(z, self.k, dim=-1)
        output = torch.zeros_like(z).scatter_(
            dim=-1,
            index=topk_indices,
            src=topk_values,
        )
        return output

    def __str__(self) -> str:
        """Return a string representation of the TopK module."""
        return f"{self.__class__.__name__}(k={self.k})"


def _all_except_last_dim(tensor: Tensor, *, keepdim: bool = False) -> Tensor:
    """Check if all elements are True along all dimensions except the last one.

    Args:
        tensor: Input boolean tensor of arbitrary shape.
        keepdim: Whether to retain reduced dimensions with size 1. Default is False.

    Returns:
        A boolean tensor containing the result of the all operation along all
        dimensions except the last one. (shape: [size of last dimension] if keepdim is
        False, else shape: [1, 1, ..., size of last dimension])

    """
    ndim = tensor.dim()
    dims_to_reduce = tuple(range(ndim - 1))
    return tensor.all(dim=dims_to_reduce, keepdim=keepdim)


def update_dead_latent_counts(activations: Tensor, prev_counts: Tensor) -> Tensor:
    """Update the count of dead latents based on the current batch activations.

    A latent is considered "dead" in a batch if for every token in the context of each
    sample its activation is zero.
    This function increments the dead latent count for each latent that is dead in the
    current batch, and resets the count to zero for latents that are active.

    Example:
        Given the following latent activations `z` for a batch of 2 samples, each with a
        context length of 3 and 5 latents:
        >>> activations = [[[0, 0, 0, 0, 5],
                            [0, 0, 3, 2, 1],
                            [0, 0, 3, 4, 5]],
                           [[0, 2, 0, 4, 5],
                            [0, 4, 3, 2, 1],
                            [0, 2, 3, 4, 5]]]
        >>> is_dead =       [1, 0, 0, 0, 0]

    Args:
        activations: The latent activations tensor. (shape [batch_size, context_len,
            num_latents]).
        prev_counts: The tensor containing previous dead neuron counts for each
            neuron. (shape: [num_latents])

    Returns:
        Updated dead neuron counts for each neuron.

    """
    # 0 is active, 1 is inactive
    dead_mask = _all_except_last_dim(activations == 0).to(dtype=torch.int)
    count = prev_counts * dead_mask  # This resets the count if the latent is active
    count += dead_mask
    return count


if __name__ == "__main__":  # pragma: no cover[demonstration-code]
    z = torch.tensor(
        [
            [
                [0, 0, 0, 0, 5],
                [0, 0, 3, 2, 1],
                [0, 0, 3, 4, 5],
            ],
            [
                [0, 2, 0, 4, 5],
                [0, 4, 3, 2, 1],
                [0, 2, 3, 4, 5],
            ],
        ],
    )
    prev_counts = torch.tensor([3, 4, 5, 6, 7])
    new_counts = update_dead_latent_counts(z, prev_counts)
    print("Updated dead latent counts:", new_counts)
    print("Expected output: tensor([4, 4, 5, 6, 7])")
