import open_clip
from configs.model_card import models
from os.path import join
import os
import gdown
import sys
import torch


def load_open_clip_model(args, device):
    """
    Model loading function for
    clip-b/32 + clip-l/14 + evaclip-l/14 + sigclip-l/16 + sigclip-b/16
    """

    model_name = models[args.model]["model"]

    if model_name not in [
        "ViT-B-32",
        "ViT-L-14",
        "EVA02-L-14",
        "ViT-B-16-SigLIP",
        "ViT-L-16-SigLIP-256",
        "ViT-SO400M-14-SigLIP-384"
    ]:
        raise ValueError(
            f"Model {model_name} not supported by Open-CLIP, please check clip loader being used."
        )

    pretrained = models[args.model]["pretrained"]
    cache_dir = args.cache_dir

    print(f"Downloading and Loading the {args.model} model")
    model, _, transform = open_clip.create_model_and_transforms(
        model_name=model_name, pretrained=pretrained, cache_dir=cache_dir, device=device
    )

    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, tokenizer, transform


def load_neg_clip_model(args, device):
    path = join(args.cache_dir, "negclip.pth")
    if not os.path.exists(path):
        print("Downloading the NegClip model...")
        gdown.download(id="1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ", output=path, quiet=False)
    model, _, transform = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=path, device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.eval()
    return model, tokenizer, transform



def load_vifi_clip_model(args, device):
    from collections import OrderedDict

    path = join(args.cache_dir, "vifi_clip_10_epochs_k400_full_finetuned.pth")
    pretrained = models[args.model]["pretrained"]
    cache_dir = args.cache_dir
    model_name = models[args.model]["model"]

    if not os.path.exists(path):
        # TODO: add script to download the model.
        print("Need to download the ViFi-CLIP model manually!")
        sys.exit(1)

    model, _, transform = open_clip.create_model_and_transforms(
        model_name=model_name, pretrained=pretrained, cache_dir=cache_dir, device=device
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    k400_checkpoint = torch.load(path, map_location="cpu")
    k400_new_sd = OrderedDict()
    for m in k400_checkpoint["model"].keys():
        new_m_name = m.replace("module.", "")
        k400_new_sd[new_m_name] = k400_checkpoint["model"][m]

    model.load_state_dict(k400_new_sd, strict=False)
    model = model.to(device)
    model = model.eval()
    return model, tokenizer, transform


def load_clip_model(args, device):
    model = args.model
    if model in [
                "clip_B_32",
                "clip_L_14",
                "evaclip_L_14",
                "siglip_B_16",
                "siglip_L_16",
            ]:
        model, tokenizer, transform = load_open_clip_model(args, device=device)
        model_type = "open_clip"

    elif model == "negclip_B_32":
        model, tokenizer, transform = load_neg_clip_model(args, device=device)
        model_type = "neg_clip"

    elif model == "vificlip":
        model, tokenizer, transform = load_vifi_clip_model(args, device=device)
        model_type = "vificlip"

    return model, tokenizer, transform, model_type


@torch.no_grad
def get_video_features(model, frames, device, model_type="open-clip"):

    if len(frames.shape) == 3:
        b, f, dim = frames.shape
        image_features = frames.to(device)  # b,20,512
        return image_features

    else:
        model.eval()
        b, f, c, h, w = frames.shape
        images = frames.reshape((b * f, c, h, w))
        images = images.to(device)

        if (
            model_type == "open_clip"
            or model_type == "neg_clip"
            or model_type == "vificlip"
        ):
            image_features = model.encode_image(images, normalize=False)

            image_features = image_features.reshape((b, f, -1)).to(device)  # b,20,512

            image_features = image_features.mean(dim=1)  # b,512
            image_features /= image_features.norm(dim=-1, keepdim=True)  # b,512

            return image_features


@torch.no_grad
def get_cap_features(model, cap, tokenizer, device, model_type="open_clip"):

    model.eval()
    if model_type == "open_clip" or model_type == "vificlip":
        text = tokenizer(cap)
        text = text.to(device)
        text_features = model.encode_text(text, normalize=True)
    elif model_type == "neg_clip":
        text = tokenizer(cap)
        text = text.to(device)
        text_features = model.encode_text(text, normalize=True)

    # TODO: Check if the feats need to be normed for all models?

    # text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.to(device)

    return text_features
