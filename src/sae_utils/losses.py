import torch
from torch import Tensor
from torch.nn import DataParallel
from torch.nn.functional import mse_loss

from sae_utils.activations import TopKActivation
from sae_utils.model import SparseAE

# Definition of loss functions for the Sparse Autoencoder (SAE)
# def loss_reconstruction(x_reconstructed: Tensor, x: Tensor) -> Tensor:
#     return F.mse_loss(x_reconstructed, x, reduction="mean")

loss_reconstruction_fn = mse_loss  # ? Why use mse_loss instead of nn.MSELoss?


def loss_k_aux(
    autoencoder: SparseAE | DataParallel[SparseAE],
    x: Tensor,
    x_reconstructed: Tensor,  # TODO: merge parameters into a dict from SAE.forward
    z_pre_activation: Tensor,
    dead_neurons_mask: Tensor,
    k_aux: int = 256,
) -> Tensor:
    # TODO: check documentation with the OpenAI article
    """Compute the auxiliary k-sparse loss for a sparse autoencoder.

    This loss measures the mean squared error (MSE) between the reconstruction error and
    its approximation using only the top-k activated neurons (after masking dead
    neurons). It encourages the autoencoder to reconstruct the input using a sparse
    subset of activations.

    Args:
        autoencoder (SparseAE | DataParallel[SparseAE]): The sparse autoencoder model
            with a decode method.
        x (Tensor): The original input tensor.
        x_reconstructed (Tensor): The reconstructed input tensor from the autoencoder.
        z_pre_activation (Tensor): The pre-activation tensor from the encoder.
        dead_neurons_mask (Tensor): A mask tensor indicating inactive (dead) neurons. A
            value of 1 indicates a dead neuron, and 0 indicates an active neuron.
        k_aux (int, optional): Number of top activations to use for the auxiliary loss.
            Defaults to 256.

    Returns:
        Tensor: The computed auxiliary k-sparse loss (MSE), with NaNs replaced by zero.

    """
    e = x - x_reconstructed
    topk_aux = TopKActivation(k=k_aux)
    z_masked = z_pre_activation * dead_neurons_mask  # TODO: check if this is correct
    e_cap = autoencoder.decode(topk_aux(z_masked))
    L2_k_aux = mse_loss(e, e_cap, reduction="mean")  # noqa: N806
    return L2_k_aux.nan_to_num(0)  # ? When would this be NaN?


def loss_top_k(
    loss_reconstruction: Tensor,
    loss_aux: Tensor,
    alpha_aux: float,
) -> Tensor:
    """Compute the combined loss for top-k selection by summing recon and aux losses.

    Args:
        loss_reconstruction (Tensor): The reconstruction loss tensor.
        loss_aux (Tensor): The auxiliary loss tensor.
        alpha_aux (float): Weighting factor for the auxiliary loss.

    Returns:
        Tensor: The combined loss as a float32 tensor.

    """
    loss = loss_reconstruction + alpha_aux * loss_aux
    return loss.to(torch.float32)
