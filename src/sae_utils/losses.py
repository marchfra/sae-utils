import torch
from torch import Tensor
from torch.nn.functional import mse_loss

from sae_utils.activations import TopK
from sae_utils.model import SAEResult, SparseAE

loss_recon_fn = mse_loss


def loss_k_aux(
    autoencoder: SparseAE,
    x: Tensor,
    sae_output: SAEResult,
    dead_latents_mask: Tensor,
    k_aux: int = 512,
) -> Tensor:
    """Compute the auxiliary k-sparse loss for a sparse autoencoder.

    This loss measures the mean squared error (MSE) between the reconstruction error and
    its approximation using only the top-k activated dead neurons. It encourages the
    autoencoder to avoid dead neurons by promoting activity in these neurons.
    For more details, see https://cdn.openai.com/papers/sparse-autoencoders.pdf,
    Sections 2.4 and A.2.

    Args:
        autoencoder: The sparse autoencoder model with a decode method.
        x: The original input tensor.
        sae_output: The output of the autoencoder's forward pass, containing:
            - latents: The post-activation latent tensor.
            - latents_pre_activation: The pre-activation latent tensor.
            - recon: The reconstructed input tensor.
            - norm: Normalization parameters used during processing.
        dead_latents_mask: A mask tensor indicating inactive (dead) neurons. A value of
            1 indicates a dead neuron, and 0 indicates an active neuron.
        k_aux: Number of top activations to use for the auxiliary loss. Defaults to 512.

    Returns:
        The computed auxiliary k-sparse loss (MSE), with NaNs replaced by zero.


    """
    topk_aux = TopK(k=k_aux)

    e = x - sae_output.recon
    dead_pre_activations = sae_output.latents_pre_activation * dead_latents_mask
    e_hat = autoencoder.decode(topk_aux(dead_pre_activations), sae_output.norm)
    return mse_loss(e, e_hat, reduction="mean").nan_to_num(0)


def loss_top_k(
    loss_reconstruction: Tensor,
    loss_aux: Tensor,
    alpha_aux: float = 1 / 32,
) -> Tensor:
    """Compute the combined loss for top-k selection by summing recon and aux losses.

    Args:
        loss_reconstruction: The reconstruction loss tensor.
        loss_aux: The auxiliary loss tensor.
        alpha_aux: Weighting factor for the auxiliary loss. Defaults to 1/32.

    Returns:
        The combined loss as a float32 tensor.

    """
    loss = loss_reconstruction + alpha_aux * loss_aux
    return loss.to(torch.float32)
