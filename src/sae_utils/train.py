from typing import Literal

import torch
from torch.nn import DataParallel, ReLU
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange

from sae_utils.activations import TopKActivation, update_dead_neuron_counts
from sae_utils.config import Config
from sae_utils.dataset import SAETrainingDataset, tied_bias_initialization
from sae_utils.losses import (
    loss_k_aux,
    loss_reconstruction_fn,
    loss_top_k,
)
from sae_utils.model import SparseAE


def train_sae(
    config: Config,
    dataset: SAETrainingDataset,
    device: Literal["cpu", "cuda"] = "cuda",
) -> tuple[SparseAE | DataParallel[SparseAE], list[float]]:
    """Train a Sparse Autoencoder (SAE) model on the provided dataset.

    Args:
        config (Config): Configuration object containing training and model
            hyperparameters.
        dataset (SAETrainingDataset): Input activations dataset.
        device (Literal["cpu", "cuda"], optional): Device to run training on. Defaults
            to "cuda".

    Returns:
        tuple:
            - SparseAE or DataParallel[SparseAE]: The trained Sparse Autoencoder model.
            - list[float]: List of loss values recorded during training.

    Notes:
        - Prints training progress and configuration details.
        - Uses Adam optimizer and supports custom activation functions.
        - Tracks dead neurons and applies auxiliary loss during training.

    """
    activation = TopKActivation(k=config.k) if config.activation == "topk" else ReLU()

    if torch.cuda.is_available() and device == "cuda":
        print("CUDA is available. Using GPU for training.")
        if torch.cuda.device_count() > 1:
            print(f"Using all {torch.cuda.device_count()} GPUs for training.")
    else:
        print("CUDA is not available or CPU specified. Using CPU for training.")
        device = "cpu"

    learning_rate = config.learning_rate
    n_epochs = config.n_epochs
    threshold_iterations_dead_latent = config.threshold_iterations_dead_latent
    alpha_aux_loss = config.alpha_aux_loss

    print("Starting SAE training")
    print(f"Using activation function: {activation}")
    print(f"Input dimension: {dataset.shape[-1]}")
    print(
        f"Latent dimension factor: {config.latent_dim_factor} | "
        f"Latent dimension: {dataset.shape[-1] * config.latent_dim_factor}",
    )
    print(f"Learning rate: {learning_rate}")
    print(f"Number of epochs: {n_epochs}")
    print(f"Device: {device}")

    autoencoder = SparseAE(
        input_dim=dataset.shape[-1],
        latent_dim_factor=config.latent_dim_factor,
        activation=activation,
    )
    if torch.cuda.device_count() > 1 and device == "cuda":
        autoencoder = DataParallel(autoencoder)
    autoencoder = autoencoder.to(device)

    autoencoder.tied_bias.data = tied_bias_initialization(dataset)  # TODO: this is ugly
    autoencoder.to(device)
    print(f"Dataset shape: {dataset.shape}")
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    print(f"Dataloader created with batch size: {config.batch_size}")
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
                k_aux=config.k_aux,
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
