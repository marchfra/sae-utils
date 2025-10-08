import argparse
import os
import pickle

import pandas as pd
import torch
from SAE import train_sae_on_activations
from scipy import sparse
from torch.utils.data import DataLoader, TensorDataset

from config import parameters  # TODO: change to json


def main():
    parser = argparse.ArgumentParser(description="Train SAE on activations")
    parser.add_argument(
        "--vision_model",
        type=str,
        default="vit",
        help="Select vision model between: vit, clip_vit, clip_resnet",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="celeba",
        help="Select dataset between: celeba, mini_imagenet, cub",
    )
    args = parser.parse_args()

    device = "cuda"
    layer = parameters["layer"]
    dim_factor = parameters["latent_dim_factor"]

    save_dir = "/data/fcozzi/sae_ontology/sae_store/"

    if args.vision_model == "clip_vit":
        load_path = f"/data/fcozzi/sae_ontology/activation_store/activations_{args.dataset}_{args.vision_model}_residual_{layer}.pickle"
    elif args.vision_model == "vit":
        load_path = f"/data/fcozzi/sae_ontology/activation_store/activations_{args.dataset}_{args.vision_model}_output_{layer}.pickle"

    hidden_activations = pd.read_pickle(load_path).squeeze()
    patch = parameters["patch"]
    if args.dataset == "tiered_imagenet":
        dataset = torch.tensor(hidden_activations).to(device)
    else:
        dataset = torch.tensor(hidden_activations)[:, patch, :].to(device)

    del hidden_activations

    trained_SAE, losses = train_sae_on_activations(
        parameters=parameters,
        dataset=dataset,
        device=device,
    )

    print("SAE training done!")

    activation_name = parameters["activation"]

    save_path = os.path.join(
        save_dir,
        f"trained_SAE_{activation_name}_{args.dataset}_{args.vision_model}_layer_{layer}_factor_{dim_factor}.pth",
    )
    torch.save(trained_SAE.state_dict(), save_path)
    with open(
        f"/data/fcozzi/sae_ontology/sae_store/SAE_training_loss_{activation_name}_{args.dataset}_{args.vision_model}_layer_{layer}_factor_{dim_factor}.pickle",
        "wb",
    ) as f:
        pickle.dump(losses, f)

    with torch.no_grad():
        # Create a DataLoader to iterate over the dataset in batches
        batch_size = 50_000  # adjust depending on memory
        dataset_loader = DataLoader(TensorDataset(dataset.cpu()), batch_size=batch_size)

        all_latents = []
        for batch in dataset_loader:
            x = batch[0].to(device, dtype=torch.float32)
            sae_output = trained_SAE(x)
            latents = sae_output["Latents"].cpu().detach()  # move back to CPU
            all_latents.append(latents)

        # Concatenate all batch results into a single tensor
        all_latents = torch.cat(all_latents, dim=0)

    # Convert to sparse
    sparse_latents = sparse.csr_matrix(all_latents.numpy())

    print("Latents done!")

    with open(
        f"/data/fcozzi/sae_ontology/latent_store/latents_{activation_name}_{args.dataset}_{args.vision_model}_layer_{layer}_factor_{dim_factor}.pickle",
        "wb",
    ) as f:
        pickle.dump(sparse_latents, f)


if __name__ == "__main__":
    main()
