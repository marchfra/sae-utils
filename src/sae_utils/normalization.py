from torch import Tensor
from torch.nn import Module


class LayerNorm(Module):
    def __init__(self, eps: float = 1e-5) -> None:
        """Initialize the object with a specified epsilon value.

        Args:
            eps (float, optional): A small value to avoid division by zero or for
                numerical stability. Defaults to 1e-5.

        """
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Normalize the input tensor `x` along its last dimension.

        Args:
            x (Tensor): Input tensor to be normalized.

        Returns:
            tuple[Tensor, Tensor, Tensor]: A tuple containing:
                - The normalized tensor.
                - The mean of `x` along the last dimension (with keepdim=True).
                - The standard deviation of `x` along the last dimension (with
                    keepdim=True).

        """
        mu = x.mean(dim=-1, keepdim=True)
        x = x - mu
        std = x.std(dim=-1, keepdim=True)
        x = x / (std + self.eps)
        return x, mu, std
