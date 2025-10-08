import torch
from geom_median.torch import compute_geometric_median


def main() -> None:
    torch.manual_seed(0)
    with torch.no_grad():
        x = torch.tensor(
            [
                [0.0, 0.0, 0.3, 0.4],
                [0.0, 0.6, 0.0, 0.8],
            ],
        )

        if torch.cuda.is_available():
            print(type(compute_geometric_median(x).median.cuda()))
        else:
            print(compute_geometric_median(x).median.dtype)


if __name__ == "__main__":
    main()
