import torch
from geom_median.torch import compute_geometric_median
from torch import Tensor
from torch.utils.data import Dataset, Subset


class SAETrainingDataset(Dataset[Tensor]):
    def __init__(self, data: Tensor) -> None:
        """Initialize the SAETrainingDataset with the provided data tensor."""
        self.data = data

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Tensor:
        """Retrieve a sample from the dataset by index."""
        return self.data[idx]

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the dataset as (num_samples, num_features)."""
        return self.data.shape


def tied_bias_initialization(
    dataset: SAETrainingDataset,
    sample_every: int = 15,
) -> Tensor:
    """Init a tied bias tensor using the geometric median of a subset of the dataset.

    Args:
        dataset (SAETrainingDataset): The training dataset containing input tensors.
        sample_every (int, optional): Interval for sampling the dataset to compute the
            geometric median. Only every `sample_every`-th sample is used to reduce
            memory usage. Defaults to 15.

    Returns:
        Tensor: The geometric median tensor, moved to CUDA if available and cast to
            float32.

    """
    subset = Subset(dataset, indices=range(0, len(dataset), sample_every))
    geom_med: Tensor = compute_geometric_median(subset).median
    if torch.cuda.is_available():
        return geom_med.cuda().to(torch.float32)
    return geom_med.to(torch.float32)
