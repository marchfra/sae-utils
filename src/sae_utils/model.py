from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import Module

from sae_utils.normalization import LayerNorm, NormalizationParams


class SAEResult(NamedTuple):
    """A NamedTuple that stores the results of a Sparse Autoencoder (SAE) forward pass.

    Attributes:
        latents: The encoded latent representations produced by the SAE.
        latents_pre_activation: The latent representations before activation is applied.
        reconstructed_input: The input reconstructed by the SAE decoder.
        norm: Parameters used for input normalization.

    """

    latents: Tensor
    latents_pre_activation: Tensor
    reconstructed_input: Tensor
    norm: NormalizationParams


class SparseAE(Module):
    """SparseAE is a PyTorch Module implementing a Sparse AutoEncoder (SAE).

    The SAE consists of an encoder and a decoder with tied weights, and includes
    preprocessing steps such as normalization. The latent space is enforced to be sparse
    using a top-k activation function.

    """

    # TODO: right now TopKActivation is not the default

    def __init__(
        self,
        input_dim: int,
        latent_dim_factor: int,
        activation: Module,
        epsilon: float = 1e-7,
    ) -> None:
        """Initialize the SparseAE (Sparse AutoEncoder) module.

        Args:
            input_dim: Dimensionality of the input features.
            latent_dim_factor: Factor to multiply input_dim to obtain latent dimension.
            activation: Activation function to use in the network.
            epsilon: Small value for numerical stability in normalization. Defaults to
                1e-7.

        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim_factor = latent_dim_factor
        self.latent_dim = self.input_dim * self.latent_dim_factor
        self.activation = activation

        self.epsilon = epsilon
        self.tied_bias = nn.Parameter(torch.zeros(self.input_dim))
        self.normalization = LayerNorm(eps=self.epsilon)
        self.lin_encoder = nn.Linear(
            in_features=self.input_dim,
            out_features=self.latent_dim,
            bias=False,
        )
        self.lin_decoder = nn.Linear(
            in_features=self.latent_dim,
            out_features=self.input_dim,
            bias=False,
        )

        # Tied weights
        self.lin_decoder.weight.data = self.lin_encoder.weight.data.T.clone()

    def encoder_pre_activation(self, x: Tensor) -> Tensor:
        """Compute the pre-activation output of the encoder.

        Before encoding, the tied bias is subtracted from the input tensor.

        Args:
            x: Input tensor to the encoder.

        Returns:
            The encoder's pre-activation output.

        """
        x = x - self.tied_bias
        z = self.lin_encoder(x)
        return z

    def decoder(self, z: Tensor, norm: NormalizationParams) -> Tensor:
        """Decode the latent representation `z` into the reconstructed input tensor.

        The decoding process involves:
        1. Passing the latent representation `z` through the decoder layer.
        2. Adding the tied bias to the decoded output.
        3. Denormalizing the result using the provided normalization parameters.

        Args:
            z: Latent representation tensor to be decoded.
            norm: NamedTuple containing normalization parameters.

        Returns:
            The reconstructed input tensor after decoding and denormalization.

        """
        x_rec = self.lin_decoder(z) + self.tied_bias
        x_rec = x_rec * (norm.std + self.epsilon) + norm.mu
        return x_rec

    def forward(self, x: Tensor) -> SAEResult:
        """Perform a forward pass through the Sparse AutoEncoder (SAE).

        This method normalizes the input, encodes it to a latent representation,
        applies the activation function, and reconstructs the input from the latent
        representation.

        Args:
            x: Input tensor to be processed.

        Returns:
            A NamedTuple (`latents`, `latents_pre_activation`, `reconstructed_input`,
                `norm`), where `latents` is the activated latent representation,
                `latents_pre_activation` is the latent representation before activation,
                `reconstructed_input` is the reconstructed input tensor, and `norm` is
                the normalization parameters used during processing.

        """
        x, norm = self.normalization(x)
        z_pre_activation = self.encoder_pre_activation(x)
        z = self.activation(z_pre_activation)
        x_reconstructed = self.decoder(z, norm)

        return SAEResult(
            latents=z,
            latents_pre_activation=z_pre_activation,
            reconstructed_input=x_reconstructed,
            norm=norm,
        )
