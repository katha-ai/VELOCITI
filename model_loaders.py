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
    
    model_name = models[args.model]['model']
    
    if model_name not in ["ViT-B-32", "ViT-L-14", "EVA02-L-14", "ViT-B-16-SigLIP", "ViT-L-16-SigLIP-256"]:
        raise ValueError(f"Model {model_name} not supported by Open-CLIP, please check clip loader being used.")
    
    pretrained = models[args.model]['pretrained']
    cache_dir = args.cache_dir
    
    print(f"Downloading and Loading the {args.model} model")
    model, _, transform = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
        cache_dir=cache_dir,
        device=device
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
    model, _, transform = open_clip.create_model_and_transforms('ViT-B-32', pretrained=path, device=device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model = model.eval()
    return model, tokenizer, transform

def load_vifi_clip_model(args, device):
    from collections import OrderedDict
    
    path = join(args.cache_dir, "vifi_clip_10_epochs_k400_full_finetuned.pth")
    pretrained = models[args.model]['pretrained']
    cache_dir = args.cache_dir
    model_name = models[args.model]['model']
    
    if not os.path.exists(path):
        #TODO: add script to download the model.
        print("Need to download the ViFi-CLIP model manually!")
        sys.exit(1)
        
    model, _, transform = open_clip.create_model_and_transforms(
        model_name = model_name,
        pretrained=pretrained,
        cache_dir=cache_dir,
        device=device)
    
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    
    k400_checkpoint = torch.load(path, map_location='cpu')
    k400_new_sd = OrderedDict()
    for m in k400_checkpoint['model'].keys():
        new_m_name = m.replace('module.', '')
        k400_new_sd[new_m_name] = k400_checkpoint['model'][m]
        
    model.load_state_dict(k400_new_sd, strict=False)
    model = model.to(device)
    model = model.eval()
    return model, tokenizer, transform


def load_clip_vip_model(args, device):
    import torch
    import requests
    from torch.nn import functional as F
    import numpy as np
    from easydict import EasyDict as edict
    import os
    os.environ['HF_HOME'] = args.cache_dir
    from transformers.models.clip.configuration_clip import CLIPConfig
    from transformers import CLIPTokenizerFast
    from transformers import AutoProcessor
    from utils.CLIP_VIP import CLIPModel
    
    extraCfg = edict({
    "type": "ViP",
    "temporal_size": 12,
    "if_use_temporal_embed": 1,
    "logit_scale_init_value": 4.60,
    "add_cls_num": 3 })
    
    if not os.path.exists(f"{args.cache_dir}/pretrain_clipvip_base_32.pt"):
        print("Downloading the CLIP-ViP model...")
        url = r"https://hdvila.blob.core.windows.net/dataset/pretrain_clipvip_base_32.pt?sp=r&st=2023-03-16T05:02:41Z&se=2027-05-31T13:02:41Z&spr=https&sv=2021-12-02&sr=b&sig=91OEG2MuszQmr16N%2Bt%2FLnvlwY3sc9CNhbyxYT9rupw0%3D"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(f"{args.cache_dir}/pretrain_clipvip_base_32.pt", 'wb') as f:
                for chunk in response.iter_content(1024):
                    if chunk:
                        f.write(chunk)
                
    
        
    clipconfig = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32", cache_dir=args.cache_dir)
    clipconfig.vision_additional_config = extraCfg
    checkpoint = torch.load(f"{args.cache_dir}/pretrain_clipvip_base_32.pt")
    cleanDict = { key.replace("clipmodel.", "") : value for key, value in checkpoint.items() }
    model =  CLIPModel(config=clipconfig)
    model.load_state_dict(cleanDict)
    model = model.to(device)
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32", cache_dir=args.cache_dir)
    processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch16", cache_dir=args.cache_dir)
    
    return model, tokenizer, processor
    


