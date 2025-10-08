import torch

"""
get_activations is a function that follows the pytorch docs to create a hook that will save the activations of a model layer.
Following the torch codes, if we want the hooks on the mlps alone, the output is a torch.Tensor:
--> Output = torch.Tensor 
However, if we want the hooks on the attns or both attns and mlps or residual blocks, the output is a tuple such as:
--> Output = (torch.Tensor, None)
The get_activations function works either way. Error raises are included to help debugging in case the output is not as expected.
"""


def get_activations(name, activation_dict):
    def hook(model, input, output):
        if isinstance(output, torch.Tensor):
            activation_dict[name] = output.detach()
        elif isinstance(output, tuple):
            if len(output) == 2:
                if output[1] is None:
                    if output[0] is not None:
                        activation_dict[name] = output[0].detach()
                    else:
                        raise (
                            "Err1: No tensor found in first element of tuple in hook, check problems in code"
                        )
                else:
                    raise (
                        "Err2: Second element of tuple of output in hook is not None, while usually it is"
                    )
            else:
                raise ("Err3: Tuple length is not 2, check problems in code")

    return hook


def initialize_hooked_vision_model(model_name, layer_list, activation_dict):
    if model_name == "CLIP_ViT-16":
        from vision_models.clip.clip import load

        model, preprocessor = load("ViT-B/16", device="cuda")
        image_model = model.visual.eval()

        for i in layer_list:
            image_model.transformer.resblocks[i].mlp.register_forward_hook(
                get_activations(f"mlp_{i}", activation_dict)
            )
            image_model.transformer.resblocks[i].attn.register_forward_hook(
                get_activations(f"attn_{i}", activation_dict)
            )
            image_model.transformer.resblocks[i].register_forward_hook(
                get_activations(f"residual_{i}", activation_dict)
            )

        return image_model, preprocessor

    if model_name == "ViT-16":
        # Initialize a Vision Transformer model and its processor
        from transformers import ViTForImageClassification, ViTImageProcessor

        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

        for i in layer_list:
            model.eval().vit.encoder.layer[i].attention.output.register_forward_hook(
                get_activations(f"attn_{i}", activation_dict)
            )
            model.eval().vit.encoder.layer[i].intermediate.register_forward_hook(
                get_activations(f"intermediate_{i}", activation_dict)
            )
            model.eval().vit.encoder.layer[i].output.register_forward_hook(
                get_activations(f"output_{i}", activation_dict)
            )

        return model, processor
