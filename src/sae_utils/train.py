from typing import Literal

import torch
from torch.nn import DataParallel, ReLU
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange

from sae_utils.activations import (
    AbsTopKActivation,
    TopKActivation,
    update_dead_neuron_counts,
)
from sae_utils.config import Config
from sae_utils.dataset import SAETrainingDataset, tied_bias_initialization
from sae_utils.losses import loss_k_aux, loss_reconstruction_fn, loss_top_k
from sae_utils.model import SparseAE


def validate_device(device: Literal["cpu", "cuda"]) -> Literal["cpu", "cuda"]:
    """Validate the specified device for training.

    Args:
        device: The device to validate.

    Returns:
        The validated device. Defaults to "cpu" if the specified device is not
            available.

    Raises:
        ValueError: If an invalid device is specified.

    """
    if device not in ["cpu", "cuda"]:
        raise ValueError(f"Invalid device specified: {device}. Choose 'cpu' or 'cuda'.")

    if device == "cuda":
        if torch.cuda.is_available():
            print("CUDA is available. Using GPU for training.")
            if torch.cuda.device_count() > 1:
                print(f"Using all {torch.cuda.device_count()} GPUs for training.")
            return "cuda"

        print("CUDA is not available. Using CPU for training.")
        return "cpu"

    print("Using CPU for training.")
    return "cpu"


def train_sae(
    config: Config,
    dataset: SAETrainingDataset,
    device: Literal["cpu", "cuda"] = "cuda",
) -> tuple[SparseAE | DataParallel[SparseAE], list[float], list[float]]:
    """Train a Sparse Autoencoder (SAE) model on the provided dataset.

    Args:
        config: Configuration object containing training and model hyperparameters.
        dataset: Input activations dataset.
        device: Device to run training on. Defaults to "cuda".

    Returns:
        A tuple (trained_sae, epoch_losses, batch_losses) where batch_losses is a list
            of losses recorded at each batch since the beginning of training.

    """
    match config.activation:
        case "topk":
            activation = TopKActivation(k=config.k)
        case "abstopk":
            activation = AbsTopKActivation(k=config.k)
        case _:
            activation = ReLU()

    device = validate_device(device)

    print("Starting SAE training")
    print(f"Using activation function: {activation}")
    print(f"Input dimension: {dataset.data.shape[-1]}")
    print(
        f"Latent dimension factor: {config.latent_dim_factor} | "
        f"Latent dimension: {dataset.data.shape[-1] * config.latent_dim_factor}",
    )
    print(f"Learning rate: {config.learning_rate}")
    print(f"Number of epochs: {config.n_epochs}")
    print(f"Device: {device}")

    autoencoder = SparseAE(
        input_dim=dataset.data.shape[-1],
        latent_dim_factor=config.latent_dim_factor,
        activation=activation,
    )
    if device == "cuda" and torch.cuda.device_count() > 1:
        autoencoder = DataParallel(autoencoder)
    autoencoder.tied_bias.data = tied_bias_initialization(
        dataset,
    )  # DataParallel has no attribute 'tied_bias'
    autoencoder.to(device)

    print(f"Dataset shape: {dataset.data.shape}")
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=config.learning_rate)
    dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=True)
    print(f"Dataloader created with batch size: {config.batch_size}")
    print(f"Number of batches: {len(dataloader)}")

    epoch_losses: list[float] = []
    batch_losses: list[float] = []
    dead_neurons_counts = torch.zeros(autoencoder.latent_dim).to(device)
    for _epochs in trange(config.n_epochs, desc="SAE training epoch"):
        for batch in tqdm(dataloader, desc="SAE training batch", leave=False):
            optimizer.zero_grad()

            x = batch.to(device=device, dtype=torch.float32)
            result = autoencoder(x)

            loss_reconstruction = loss_reconstruction_fn(
                input=result.reconstructed_input,
                target=x,
            )

            # Update dead neurons counts and create mask
            dead_neurons_counts = update_dead_neuron_counts(
                z=result.latents.detach(),
                prev_counts=dead_neurons_counts.clone(),
            )
            dead_neurons_mask = dead_neurons_counts > config.threshold_dead_latent

            aux_loss = loss_k_aux(
                autoencoder=autoencoder,
                x=x,
                sae_output=result,
                dead_neurons_mask=dead_neurons_mask,
                k_aux=config.k_aux,
            )
            batch_loss = loss_top_k(
                loss_reconstruction=loss_reconstruction,
                loss_aux=aux_loss,
                alpha_aux=config.alpha_aux_loss,
            )

            batch_losses.append(batch_loss.item())
            batch_loss.backward()
            optimizer.step()

        epoch_losses.append(sum(batch_losses) / len(batch_losses))

    return autoencoder, epoch_losses, batch_losses
