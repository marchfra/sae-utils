from typing import NamedTuple

import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm.auto import trange

from sae_utils.activations import TopK, update_dead_latent_counts
from sae_utils.config import Config
from sae_utils.dataset import SAEDataset, compute_tied_bias
from sae_utils.losses import loss_k_aux, loss_recon_fn, loss_top_k
from sae_utils.model import SAEResult, SparseAE


class _SAETrainingOutput(NamedTuple):
    """Named tuple describing the outputs of a Sparse Autoencoder training run.

    Attributes
    ----------
    sae : SparseAE
        The trained Sparse Autoencoder instance returned after training.
    epoch_train_losses : list[float]
        Training loss recorded at the end of each epoch.
    batch_train_losses : list[float]
        Training loss recorded for each batch across all epochs.
    epoch_val_losses : list[float]
        Validation loss recorded at the end of each epoch.
    best_epoch : int
        Index (0-based) of the epoch with the lowest validation loss.

    """

    sae: DataParallel[SparseAE]
    epoch_train_losses: list[float]
    batch_train_losses: list[float]
    epoch_val_losses: list[float]
    best_epoch: int


def train_sae(
    config: Config,
    train_set: SAEDataset,
    val_set: SAEDataset,
    *,
    verbose: bool = True,
) -> _SAETrainingOutput:
    """Train a Sparse Autoencoder (SAE) model on the provided dataset.

    After each epoch, the model is evaluated on the validation set. The model with the
    lowest validation loss at the end of training is returned.

    Args:
        config: Configuration object containing training and model hyperparameters.
        train_set: Input activations dataset.
        val_set: Validation dataset.
        verbose: Whether to print training information. Defaults to True.

    Returns:
        A _SAETrainingOutput named tuple containing the best trained SAE model, training
        losses per epoch, training losses per batch, validation losses per epoch, and
        the epoch number with the best validation loss.

    """
    activation = TopK(k=config.k)

    # Create data loaders
    train_loader, val_loader = _create_data_loaders(
        train_set=train_set,
        val_set=val_set,
        batch_size=config.batch_size,
    )

    # Print training information
    if verbose:
        _print_training_info(
            config=config,
            train_set=train_set,
            activation=activation,
            n_batches=len(train_loader),
        )

    # Initialize model
    sae = _initialize_sae(
        train_set=train_set,
        config=config,
        activation=activation,
    )

    optimizer = torch.optim.Adam(sae.module.parameters(), lr=config.learning_rate)

    # Initialize tracking variables
    epoch_train_losses: list[float] = []
    epoch_val_losses: list[float] = []
    batch_train_losses: list[float] = []
    dead_neurons_counts = torch.zeros(sae.module.latent_dim, dtype=torch.long).to(
        config.device,
    )
    best_val_loss = float("inf")
    best_sae_state_dict = sae.module.state_dict()
    best_epoch = -1

    # Training loop
    for epoch in trange(config.n_epochs, desc="SAE training", unit="epoch"):
        # Train for one epoch
        epoch_batch_losses, dead_neurons_counts = _train_one_epoch(
            sae=sae,
            optimizer=optimizer,
            train_loader=train_loader,
            dead_neurons_counts=dead_neurons_counts,
            config=config,
        )
        batch_train_losses.extend(epoch_batch_losses)
        epoch_train_losses.append(sum(epoch_batch_losses) / len(epoch_batch_losses))

        # Validate
        epoch_val_loss = _validate_epoch(
            sae=sae,
            val_loader=val_loader,
            dead_neurons_counts=dead_neurons_counts,
            config=config,
        )
        epoch_val_losses.append(epoch_val_loss)

        # Save best model based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_sae_state_dict = sae.module.state_dict()
            best_epoch = epoch

    # Load best model
    sae.module.load_state_dict(best_sae_state_dict)

    return _SAETrainingOutput(
        sae,
        epoch_train_losses,
        batch_train_losses,
        epoch_val_losses,
        best_epoch,
    )


def _print_training_info(
    config: Config,
    train_set: SAEDataset,
    activation: TopK,
    n_batches: int,
) -> None:
    """Print training configuration information."""
    print("Starting SAE training")
    print(f"Using activation function: {activation}")
    print(f"Input shape: {train_set.data.shape}")
    latent_shape = (
        *train_set.data.shape[:-1],
        train_set.data.shape[-1] * config.latent_dim_factor,
    )
    print(
        f"Latent dimension factor: {config.latent_dim_factor} "
        f"| Latent shape: {latent_shape}",
    )
    print(f"Number of epochs: {config.n_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Device: {config.device}")
    print(
        f"Dataloader created with batch size: {config.batch_size} "
        f"| Number of batches: {n_batches}",
    )


def _create_data_loaders(
    train_set: SAEDataset,
    val_set: SAEDataset,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


def _initialize_sae(
    train_set: SAEDataset,
    config: Config,
    activation: TopK,
) -> DataParallel[SparseAE]:
    """Initialize the Sparse Autoencoder model."""
    sae = SparseAE(
        input_dim=train_set.data.shape[-1],
        latent_dim_factor=config.latent_dim_factor,
        activation=activation,
    )
    sae.init_tied_bias(compute_tied_bias(train_set))
    sae = DataParallel(sae)
    sae.to(config.device)
    sae.normalization.eps = sae.normalization.eps.to(config.device)
    return sae


def _compute_batch_loss(
    sae: SparseAE,
    sae_output: SAEResult,
    x: torch.Tensor,
    dead_neurons_mask: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    """Compute the loss for a single batch."""
    loss_recon = loss_recon_fn(input=sae_output.recon, target=x)

    aux_loss = loss_k_aux(
        autoencoder=sae,
        x=x,
        sae_output=sae_output,
        dead_latents_mask=dead_neurons_mask,
        k_aux=config.k_aux,
    )
    batch_loss = loss_top_k(
        loss_reconstruction=loss_recon,
        loss_aux=aux_loss,
        alpha_aux=config.alpha_aux_loss,
    )

    return batch_loss


def _train_one_epoch(
    sae: DataParallel[SparseAE],
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    dead_neurons_counts: torch.Tensor,
    config: Config,
) -> tuple[list[float], torch.Tensor]:
    """Train the SAE for one epoch.

    Returns:
        Tuple of (batch_losses, updated_dead_neurons_counts).

    """
    sae.train()
    batch_losses: list[float] = []

    for batch in train_loader:
        optimizer.zero_grad()

        x = batch.to(device=config.device, dtype=torch.float32)
        result = sae(x)

        # Update dead neurons counts and create mask
        dead_neurons_counts = update_dead_latent_counts(
            z=result.latents.detach(),
            prev_counts=dead_neurons_counts.clone(),
        )
        dead_neurons_mask = dead_neurons_counts > config.threshold_dead_latent

        batch_loss = _compute_batch_loss(
            sae=sae.module,
            sae_output=result,
            x=x,
            dead_neurons_mask=dead_neurons_mask,
            config=config,
        )

        batch_losses.append(batch_loss.item())
        batch_loss.backward()
        optimizer.step()

    return batch_losses, dead_neurons_counts


def _validate_epoch(
    sae: DataParallel[SparseAE],
    val_loader: DataLoader,
    dead_neurons_counts: torch.Tensor,
    config: Config,
) -> float:
    """Validate the SAE for one epoch.

    Returns:
        Average validation loss for the epoch.

    """
    sae.eval()
    val_losses: list[float] = []

    with torch.no_grad():
        for val_batch in val_loader:
            x_val = val_batch.to(device=config.device, dtype=torch.float32)
            val_result = sae(x_val)

            # TODO: Figure out how to handle dead neurons in validation
            # Create dead latents mask for validation
            dead_neurons_mask_val = dead_neurons_counts > config.threshold_dead_latent

            val_batch_loss = _compute_batch_loss(
                sae=sae.module,
                sae_output=val_result,
                x=x_val,
                dead_neurons_mask=dead_neurons_mask_val,
                config=config,
            )
            val_losses.append(val_batch_loss.item())

    return sum(val_losses) / len(val_losses)
