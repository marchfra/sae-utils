from typing import NamedTuple

from torch import Tensor
from torch.nn import Module


class NormParams(NamedTuple):
    """A NamedTuple that stores normalization parameters for data preprocessing.

    Attributes:
        mu: The mean values used for normalization.
        std: The standard deviation values used for normalization.

    """

    mu: Tensor
    std: Tensor


class LayerNorm(Module):
    def __init__(self, eps: float = 1e-5) -> None:
        """Initialize the object with a specified epsilon value.

        Args:
            eps: A small value to avoid division by zero. Defaults to 1e-5.

        """
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> tuple[Tensor, NormParams]:
        """Normalize the input tensor `x` along its last dimension.

        Args:
            x: Input tensor to be normalized.

        Returns:
            A tuple (normalized_tensor, norm_params), where `normalized_tensor` is the
                normalized version of `x`, and `norm_params` is a NamedTuple containing
                the mean and standard deviation of `x` along the last dimension.

        """
        mu = x.mean(dim=-1, keepdim=True)
        x = x - mu
        std = x.std(dim=-1, keepdim=True)
        x = x / (std + self.eps)
        return x, NormParams(mu, std)
