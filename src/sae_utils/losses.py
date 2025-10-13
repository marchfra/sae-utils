from typing import Literal

import torch
from torch import Tensor
from torch.nn import DataParallel
from torch.nn.functional import mse_loss

from sae_utils.activations import AbsTopKActivation, TopKActivation
from sae_utils.model import SAEResult, SparseAE

loss_reconstruction_fn = mse_loss


def loss_k_aux(  # noqa: PLR0913
    autoencoder: SparseAE | DataParallel[SparseAE],
    x: Tensor,
    sae_output: SAEResult,
    dead_neurons_mask: Tensor,
    k_aux: int = 512,
    activation: Literal["topk", "abstopk"] = "topk",
) -> Tensor:  # pragma: no cover[taken-from-trusted-source]
    """Compute the auxiliary k-sparse loss for a sparse autoencoder.

    This loss measures the mean squared error (MSE) between the reconstruction error and
    its approximation using only the top-k activated neurons (after masking dead
    neurons). It encourages the autoencoder to reconstruct the input using a sparse
    subset of activations.
    For more details, see https://cdn.openai.com/papers/sparse-autoencoders.pdf,
    Sections 2.4 and A.2.

    Args:
        autoencoder: The sparse autoencoder model with a decode method.
        x: The original input tensor.
        sae_output: The output of the autoencoder's forward pass, containing:
            - latents: The post-activation latent tensor.
            - latents_pre_activation: The pre-activation latent tensor.
            - reconstructed_input: The reconstructed input tensor.
            - norm: Normalization parameters used during processing.
        dead_neurons_mask: A mask tensor indicating inactive (dead) neurons. A value of
            1 indicates a dead neuron, and 0 indicates an active neuron.
        k_aux: Number of top activations to use for the auxiliary loss. Defaults to 512.
        activation: Type of top-k activation to use. "topk" selects the k largest
            activations, while "abstopk" selects the k activations with the largest
            absolute values. Defaults to "topk".

    Returns:
        The computed auxiliary k-sparse loss (MSE), with NaNs replaced by zero.

    Raises:
        ValueError: If the specified loss type is not "topk" or "abstopk".

    """
    if activation == "topk":
        topk_aux = TopKActivation(k=k_aux)
    elif activation == "abstopk":
        topk_aux = AbsTopKActivation(k=k_aux)
    else:
        raise ValueError(
            f"Invalid activation type: {activation}. Choose 'topk' or 'abstopk'.",
        )

    e = x - sae_output.reconstructed_input
    dead_pre_activations = sae_output.latents_pre_activation * dead_neurons_mask
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        e_hat = autoencoder.module.decoder(
            topk_aux(dead_pre_activations),
            sae_output.norm,
        )
    else:
        e_hat = autoencoder.decoder(topk_aux(dead_pre_activations), sae_output.norm)
    return mse_loss(e, e_hat, reduction="mean").nan_to_num(0)


def loss_top_k(
    loss_reconstruction: Tensor,
    loss_aux: Tensor,
    alpha_aux: float,
) -> Tensor:  # pragma: no cover[simple-function]
    """Compute the combined loss for top-k selection by summing recon and aux losses.

    Args:
        loss_reconstruction: The reconstruction loss tensor.
        loss_aux: The auxiliary loss tensor.
        alpha_aux: Weighting factor for the auxiliary loss.

    Returns:
        The combined loss as a float32 tensor.

    """
    loss = loss_reconstruction + alpha_aux * loss_aux
    return loss.to(torch.float32)
