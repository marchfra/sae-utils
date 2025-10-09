from typing import Any, Literal

import torch
from geom_median.torch import compute_geometric_median
from torch import Tensor, nn
from torch.nn import Module
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange


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


def tied_bias_initialization(dataset: Tensor, sample_every: int = 15) -> Tensor:
    """Init a tied bias tensor using the geometric median of a subset of the dataset.

    Args:
        dataset (Tensor): The input dataset tensor of shape (num_samples, num_features).
        sample_every (int, optional): The interval at which samples are taken from the
            dataset for geometric median computation. The dataset is downsampled to
            avoid memory issues. Defaults to 15.

    Returns:
        Tensor: The geometric median tensor, moved to CUDA if available and cast to
            float.

    """
    geom_med = compute_geometric_median(
        dataset[::sample_every, :].float().cpu(),  # ? Why .float().cpu()?
    ).median
    if torch.cuda.is_available():
        return geom_med.cuda().float()
    return geom_med.float()


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
                normalization. Default is 1e-7.

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


# Definition of loss functions for the Sparse Autoencoder (SAE)
# def loss_reconstruction(x_reconstructed: Tensor, x: Tensor) -> Tensor:
#     return F.mse_loss(x_reconstructed, x, reduction="mean")

loss_reconstruction_fn = mse_loss  # ? Why use mse_loss instead of nn.MSELoss?


def loss_k_aux(
    autoencoder: SparseAE,
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
    neurons).  It encourages the autoencoder to reconstruct the input using a sparse
    subset of activations.

    Args:
        autoencoder (SparseAE): The sparse autoencoder model with a decode method.
        x (Tensor): The original input tensor.
        x_reconstructed (Tensor): The reconstructed input tensor from the autoencoder.
        z_pre_activation (Tensor): The pre-activation tensor from the encoder.
        dead_neurons_mask (Tensor): A mask tensor indicating inactive (dead) neurons. A
            value of 1 indicates a dead neuron, and 0 indicates an active neuron.
        k_aux (int, optional): Number of top activations to use for the auxiliary loss.
            Default is 256.

    Returns:
        Tensor: The computed auxiliary k-sparse loss (MSE), with NaNs replaced by zero.

    """
    e = x - x_reconstructed
    topk_aux = TopKActivation(k=k_aux)
    z_masked = z_pre_activation * dead_neurons_mask  # TODO: check if this is correct
    e_cap = autoencoder.decode(topk_aux(z_masked))
    L2_k_aux = F.mse_loss(e, e_cap, reduction="mean")  # noqa: N806
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


def train_sae_on_activations(
    parameters: dict[
        str,
        Any,
    ],  # TODO: better define parameters, e.g., with Dataclass or TypedDict
    dataset: Tensor,
    device: Literal["cpu", "cuda"] = "cuda",
) -> tuple[SparseAE, list[float]]:
    """Train a Sparse Autoencoder (SAE) model on the provided dataset.

    Args:
        parameters (dict): Dictionary containing training and model hyperparameters.
            Expected keys include:
                - "activation": Activation function type ("topk" or other).
                - "k": Number of top activations to keep (if using "topk").
                - "learning_rate": Learning rate for optimizer.
                - "n_epochs": Number of training epochs.
                - "threshold_iterations_dead_latent": Threshold for dead neuron
                    detection.
                - "alpha_aux_loss": Weight for auxiliary loss.
                - "latent_dim_factor": Factor to determine latent dimension size.
                - "batch_size": Batch size for training.
                - "k_aux": Parameter for auxiliary loss.
        dataset (Tensor): Input activations dataset of shape (num_samples, input_dim).
        device (Literal["cpu", "cuda"], optional): Device to run training on. Default is
            "cuda".

    Returns:
        tuple:
            - SparseAE: The trained Sparse Autoencoder model.
            - list[float]: List of loss values recorded during training.

    Notes:
        - The function prints training progress and configuration details.
        - Uses Adam optimizer and supports custom activation functions.
        - Tracks dead neurons and applies auxiliary loss during training.

    """
    if parameters["activation"] == "topk":
        activation = TopKActivation(k=parameters["k"])
    else:
        activation = nn.ReLU()

    if torch.cuda.is_available() and device == "cuda":
        print("CUDA is available. Using GPU for training.")
        if torch.cuda.device_count() > 1:
            print(f"Using all {torch.cuda.device_count()} GPUs for training.")
    else:
        print("CUDA is not available or CPU specified. Using CPU for training.")
        device = "cpu"

    learning_rate = parameters["learning_rate"]
    n_epochs = parameters["n_epochs"]
    threshold_iterations_dead_latent = parameters["threshold_iterations_dead_latent"]
    alpha_aux_loss = parameters["alpha_aux_loss"]

    print("Starting SAE training")
    print(f"Using activation function: {activation}")
    print(f"Input dimension: {dataset.shape[-1]}")
    print(
        f"Latent dimension factor: {parameters['latent_dim_factor']} | "
        f"Latent dimension: {dataset.shape[-1] * parameters['latent_dim_factor']}",
    )
    print(f"Learning rate: {learning_rate}")
    print(f"Number of epochs: {n_epochs}")
    print(f"Device: {device}")

    autoencoder = SparseAE(
        input_dim=dataset.shape[-1],
        latent_dim_factor=parameters["latent_dim_factor"],
        activation=activation,
    )
    if torch.cuda.device_count() > 1 and device == "cuda":
        autoencoder = nn.DataParallel(autoencoder)
    autoencoder = autoencoder.to(device)

    autoencoder.tied_bias.data = tied_bias_initialization(dataset)  # TODO: this is ugly
    autoencoder.to(device)
    print(f"Dataset shape: {dataset.shape}")
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    dataloader = DataLoader(
        dataset=dataset,  # TODO: this might technically work, but it'd be better to use a torch.utils.data.Dataset
        batch_size=parameters["batch_size"],
        shuffle=True,
    )
    print(f"Dataloader created with batch size: {parameters['batch_size']}")
    print(f"Number of batches: {len(dataloader)}")

    losses: list[float] = []
    dead_neurons_counts = torch.zeros(autoencoder.latent_dim).to(device)
    for _ in trange(n_epochs, desc="SAE training epoch"):
        for batch in tqdm(dataloader, desc="SAE training batch", leave=False):
            optimizer.zero_grad()

            x = batch.to(device=device, dtype=torch.float32)
            # ? Why not call autoencoder.forward(x)?
            x_norm, norm_dict = autoencoder.preprocessing(x)
            z_pre_activation = autoencoder.encoder_pre_activation(x_norm)
            z = autoencoder.activation(z_pre_activation)
            x_reconstructed = autoencoder.decode(z, norm_dict)
            loss_reconstruction = loss_reconstruction_fn(
                input=x_reconstructed,
                target=x,
            )

            dead_neurons_counts = update_dead_neuron_counts(
                z=z.detach(),
                prev_counts=dead_neurons_counts.clone(),
            )
            dead_neurons_mask = dead_neurons_counts > threshold_iterations_dead_latent

            loss_aux = loss_k_aux(
                autoencoder=autoencoder,
                x=x,
                x_reconstructed=x_reconstructed,
                z_pre_activation=z_pre_activation,
                dead_neurons_mask=dead_neurons_mask,
                k_aux=parameters["k_aux"],
            )
            loss = loss_top_k(
                loss_reconstruction=loss_reconstruction,
                loss_aux=loss_aux,
                alpha_aux=alpha_aux_loss,
            )

            losses.append(loss.item())
            loss.backward()
            optimizer.step()

    return autoencoder, losses
