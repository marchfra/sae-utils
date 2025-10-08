import torch
import torch.nn.functional as F  # noqa: N812
from geom_median.torch import compute_geometric_median
from torch import Tensor, nn
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm


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


# Selected initialization for tied bias every 15 samples to avoid memory issues
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


# Define the Sparse Autoencoder (SAE) class
class SparseAE(Module):
    def __init__(self, input_dim: int, latent_dim_factor: int, activation) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim_factor = latent_dim_factor
        self.latent_dim = self.input_dim * self.latent_dim_factor
        self.activation = activation

        self.eps = 1e-7
        self.tied_bias = nn.Parameter(torch.zeros(input_dim))
        self.normalization = LayerNorm(eps=self.eps)
        self.encode = nn.Linear(
            in_features=self.input_dim,
            out_features=self.latent_dim,
            bias=False,
        )
        self.decode = nn.Linear(
            in_features=self.latent_dim,
            out_features=self.input_dim,
            bias=False,
        )

        self.decode.weight.data = self.encode.weight.data.T.clone()

    def preprocessing(self, x: Tensor):
        x, mu, std = self.normalization(x)
        normalization_dict = {"mu": mu, "std": std}
        return x, normalization_dict

    def encoder_pre_activation(self, x: Tensor):
        x = x - self.tied_bias
        x = self.encode(x)
        return x

    def decoder(self, z, normalization_dict):
        x_rec = self.decode(z) + self.tied_bias
        x_rec = (
            x_rec * (normalization_dict["std"] + self.eps) + normalization_dict["mu"]
        )
        return x_rec

    def forward(self, x: Tensor):
        x, normalization_dict = self.preprocessing(x)
        z_pre_activation = self.encoder_pre_activation(x)
        z = self.activation(z_pre_activation)
        x_reconstructed = self.decoder(z, normalization_dict)

        output = {
            "Latents": z,
            "Latents pre-activation": z_pre_activation,
            "Reconstructed input": x_reconstructed,
        }

        return output


# Definition of loss functions for the Sparse Autoencoder (SAE)
def Loss_reconstruction(x_reconstructed, x):
    return F.mse_loss(x_reconstructed, x, reduction="mean")


def Loss_k_aux(
    autoencoder,
    x,
    x_reconstructed,
    z_pre_activation,
    dead_neurons_mask,
    k_aux=256,
):
    e = x - x_reconstructed
    topk_aux = TopKActivation(k=k_aux)
    z_masked = z_pre_activation * dead_neurons_mask
    e_cap = autoencoder.decode(topk_aux(z_masked))
    L2_k_aux = F.mse_loss(e, e_cap, reduction="mean")
    return L2_k_aux.nan_to_num(0)


def LossTopK(loss_reconstruction, loss_aux, alpha_aux):
    loss = loss_reconstruction + alpha_aux * loss_aux
    return loss.to(torch.float32)


# Function to train the Sparse Autoencoder (SAE) on activations
def train_sae_on_activations(parameters, dataset, device):
    activation = TopKActivation(k=parameters["k"])

    learning_rate = parameters["learning_rate"]
    n_epochs = parameters["n_epochs"]

    print("Starting SAE training")
    print(f"Using activation function: {parameters['activation']}")
    print(f"Input dimension: {dataset.shape[-1]}")
    print(f"Latent dimension factor: {parameters['latent_dim_factor']}")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of epochs: {n_epochs}")
    print(f"Device: {device}")

    autoencoder = SparseAE(
        input_dim=dataset.shape[-1],
        latent_dim_factor=parameters["latent_dim_factor"],
        activation=activation,
    ).to(device)

    autoencoder.tied_bias.data = tied_bias_initialization(dataset)
    autoencoder.to(device)
    print(dataset.shape)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=parameters["batch_size"],
        shuffle=True,
    )
    print("Dataloader created with batch size: ", parameters["batch_size"])
    print(f"Number of batches: {len(dataloader)}")

    losses = []
    dead_neurons_counting = torch.zeros(autoencoder.latent_dim).to(device)
    for _ in tqdm(range(n_epochs), desc="SAE training epoch"):
        for batch in dataloader:
            optimizer.zero_grad()

            x = batch.to(torch.float32).to(device)
            x_norm, norm_dict = autoencoder.preprocessing(x)
            z_pre_activation = autoencoder.encoder_pre_activation(x_norm)
            z = autoencoder.activation(z_pre_activation)
            x_reconstructed = autoencoder.decoder(z, norm_dict)
            loss_reconstruction = Loss_reconstruction(
                x_reconstructed=x_reconstructed,
                x=x,
            )

            threshold_iterations_dead_latent = parameters[
                "threshold_iterations_dead_latent"
            ]
            alpha_aux_loss = parameters["alpha_aux_loss"]
            dead_neurons_counting = update_dead_neuron_counts(
                z=z.detach(),
                prev_counts=dead_neurons_counting.clone(),
            )
            dead_neurons_mask = dead_neurons_counting > threshold_iterations_dead_latent

            loss_aux = Loss_k_aux(
                autoencoder=autoencoder,
                x=x,
                x_reconstructed=x_reconstructed,
                z_pre_activation=z_pre_activation,
                dead_neurons_mask=dead_neurons_mask,
                k_aux=parameters["k_aux"],
            )
            loss = LossTopK(
                loss_reconstruction=loss_reconstruction,
                loss_aux=loss_aux,
                alpha_aux=alpha_aux_loss,
            )

            losses.append(loss.item())
            loss.backward()
            optimizer.step()

    return autoencoder, losses
