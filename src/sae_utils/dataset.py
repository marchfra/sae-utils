import torch
from geom_median.torch import compute_geometric_median
from torch import Tensor
from torch.utils.data import Dataset, Subset


class SAEDataset(Dataset[Tensor]):
    """A PyTorch Dataset for training Sparse Autoencoders (SAE) using tensor data."""

    def __init__(self, data: Tensor) -> None:
        """Initialize the SAETrainingDataset with the provided data tensor."""
        self.data = data

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Tensor:
        """Retrieve a sample from the dataset by index."""
        return self.data[idx]


def compute_tied_bias(
    dataset: SAEDataset,
    sample_every: int = 15,
) -> Tensor:
    """Init a tied bias tensor using the geometric median of a subset of the dataset.

    Args:
        dataset: The training dataset containing input tensors.
        sample_every: Interval for sampling the dataset to compute the
            geometric median. Only every `sample_every`-th sample is used to reduce
            memory usage. Defaults to 15.

    Returns:
        The geometric median tensor, moved to CUDA if available and cast to float32.

    """
    subset = Subset(dataset, indices=range(0, len(dataset), sample_every))
    last_dim = subset[:].shape[-1]
    geom_med: Tensor = compute_geometric_median(subset[:].reshape(-1, last_dim)).median
    if torch.cuda.is_available():
        geom_med = geom_med.cuda()
    return geom_med.to(torch.float32)


if __name__ == "__main__":
    dataset = SAEDataset(torch.randn(100, 5))
    subset = Subset(dataset, indices=range(0, len(dataset), 9))
    print("Dataset samples:", dataset.data.shape)
    print("Subset samples:", subset[:].shape)

    last_dim = subset[:].shape[-1]
    median = compute_geometric_median(subset[:].reshape(-1, last_dim)).median
    print("Geometric median shape:", median.shape)
