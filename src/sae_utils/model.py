import torch
from torch import Tensor, nn
from torch.nn import Module

from sae_utils.normalization import LayerNorm


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
            input_dim (int): Dimensionality of the input features.
            latent_dim_factor (int): Factor to multiply input_dim to obtain latent
                dimension.
            activation (Module): Activation function to use in the network.
            epsilon (float, optional): Small value for numerical stability in
                normalization. Defaults to 1e-7.

        Attributes:
            input_dim (int): Dimensionality of the input features.
            latent_dim_factor (int): Factor for latent dimension calculation.
            latent_dim (int): Dimensionality of the latent space.
            activation (Module): Activation function.
            epsilon (float): Numerical stability constant.
            tied_bias (nn.Parameter): Bias parameter tied to the input dimension.
            normalization (LayerNorm): Layer normalization module.
            encoder (nn.Linear): Linear encoder layer.
            decode (nn.Linear): Linear decoder layer with tied weights.

        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim_factor = latent_dim_factor
        self.latent_dim = self.input_dim * self.latent_dim_factor
        self.activation = activation

        self.epsilon = epsilon
        self.tied_bias = nn.Parameter(torch.zeros(self.input_dim))
        self.normalization = LayerNorm(eps=self.epsilon)
        self.encoder = nn.Linear(
            in_features=self.input_dim,
            out_features=self.latent_dim,
            bias=False,
        )
        self.decode = nn.Linear(
            in_features=self.latent_dim,
            out_features=self.input_dim,
            bias=False,
        )

        # Tied weights
        self.decode.weight.data = self.encoder.weight.data.T.clone()

    def preprocessing(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Normalize the input tensor and return normalization parameters."""
        x, mu, std = self.normalization(x)
        normalization = {"mu": mu, "std": std}
        return x, normalization

    def encoder_pre_activation(self, x: Tensor) -> Tensor:
        """Compute the pre-activation output of the encoder.

        Before encoding, the tied bias is subtracted from the input tensor.

        Args:
            x (Tensor): Input tensor to the encoder.

        Returns:
            Tensor: The encoder's pre-activation output.

        """
        x = x - self.tied_bias
        z = self.encoder(x)
        return z

    def decoder(self, z: Tensor, normalization: dict[str, Tensor]) -> Tensor:
        """Decode the latent representation `z` into the reconstructed input tensor.

        The decoding process involves passing the latent representation `z` through the
        decoder layer, adding the tied bias, and then denormalizing the result using
        the provided normalization parameters.

        Args:
            z (Tensor): Latent representation tensor to be decoded.
            normalization (dict[str, Tensor]): Dictionary containing normalization
                parameters:
                - "mu": Mean tensor for denormalization.
                - "std": Standard deviation tensor for denormalization.

        Returns:
            Tensor: The reconstructed input tensor after decoding and denormalization.

        """
        x_rec = self.decode(z) + self.tied_bias
        x_rec = x_rec * (normalization["std"] + self.epsilon) + normalization["mu"]
        return x_rec

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Perform a forward pass through the SAE.

        Args:
            x (Tensor): Input tensor to be processed.

        Returns:
            dict[str, Tensor]: A dictionary containing:
                - "Latents": The latent representation after activation.
                - "Latents pre-activation": The latent representation before activation.
                - "Reconstructed input": The reconstructed input tensor.

        """
        x, normalization = self.preprocessing(x)
        z_pre_activation = self.encoder_pre_activation(x)
        z = self.activation(z_pre_activation)
        x_reconstructed = self.decode(z, normalization)

        output = {
            "Latents": z,
            "Latents pre-activation": z_pre_activation,
            "Reconstructed input": x_reconstructed,
        }

        return output
