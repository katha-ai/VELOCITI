# [Minimal Working Codes to Run Various CLIP Models from Various Sources]

# ------------------------------------------------------------------------------------------
# OPEN-AI CLIP Models [https://github.com/openai/CLIP] #
# ViT-B/32 | ViT-L/14
# Installation: pip install git+https://github.com/openai/CLIP.git
# ------------------------------------------------------------------------------------------

import torch
import clip
from PIL import Image
cache_dir = r'/data'

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, download_root=cache_dir)
#or
model, preprocess = clip.load("VIT-L/14", device=device, download_root=cache_dir)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    
# ------------------------------------------------------------------------------------------
# OPEN-CLIP Models [https://github.com/mlfoundations/open_clip] #
# EvaCLIP-L/14
# Installation: pip install open_clip_torch
# ------------------------------------------------------------------------------------------

import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14',
                                                             pretrained='timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k',
                                                             device = device,
                                                             cache_dir=cache_dir)

model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-L-14')

image = preprocess(Image.open("docs/CLIP.png")).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]


# ------------------------------------------------------------------------------------------
# Google Models [https://github.com/merveenoyan/siglip] #
# SigLip-B/16 | SigLip-L/16
# https://huggingface.co/google/siglip-base-patch16-224 | https://huggingface.co/google/siglip-large-patch16-256
# Installation: pip install transformers
# ------------------------------------------------------------------------------------------

from PIL import Image
import requests
import os
os.environ["HF_HOME"] = cache_dir
from transformers import AutoProcessor, AutoModel
import torch

model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

# or

model = AutoModel.from_pretrained("google/siglip-large-patch16-256")
processor = AutoProcessor.from_pretrained("google/siglip-large-patch16-256")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

texts = ["a photo of 2 cats", "a photo of 2 dogs"]
inputs = processor(text=texts, images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image) # these are the probabilities
print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")


# ------------------------------------------------------------------------------------------
# Open-CLIP Model [https://github.com/vinid/neg_clip]
# NegCLIP-B/32
# https://drive.google.com/uc?id=1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ&confirm=t&uuid=69f059ee-b20a-4df5-9ba6-5476c3a5c1d6
# or
# https://drive.google.com/uc?id=1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ
# Installation: pip install gdown
# Inspired from ARO: https://github.com/mertyg/vision-language-models-are-bows/blob/main/model_zoo/__init__.py#L66
# ------------------------------------------------------------------------------------------

import open_clip

path = os.path.join(cache_dir, "negclip.pth")
if not os.path.exists(path):
    print(f"Downloading the NegCLIP model to path {path}")
    import gdown
    gdown.download(id="1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ", output=path, quiet=False)
    
model, _, image_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=path, device=device)
model = model.eval()


# ------------------------------------------------------------------------------------------
# CLIP-VIP [https://github.com/microsoft/XPretrain/tree/main/CLIP-ViP]
# CLIP-VIP-B32
# Model download from --> https://hdvila.blob.core.windows.net/dataset/pretrain_clipvip_base_32.pt?sp=r&st=2023-03-16T05:02:41Z&se=2027-05-31T13:02:41Z&spr=https&sv=2021-12-02&sr=b&sig=91OEG2MuszQmr16N%2Bt%2FLnvlwY3sc9CNhbyxYT9rupw0%3D
# Installation: https://github.com/microsoft/XPretrain/blob/main/CLIP-ViP/src/modeling/CLIP_ViP.py
# ------------------------------------------------------------------------------------------


import torch
from torch.nn import functional as F
import numpy as np
from easydict import EasyDict as edict

from transformers.models.clip.configuration_clip import CLIPConfig
from transformers import CLIPTokenizerFast
from transformers import AutoProcessor
from utils.CLIP_VIP import CLIPModel

extraCfg = edict({
    "type": "ViP",
    "temporal_size": 12,
    "if_use_temporal_embed": 1,
    "logit_scale_init_value": 4.60,
    "add_cls_num": 3
})

clipconfig = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dir)
clipconfig.vision_additional_config = extraCfg

checkpoint = torch.load(f"{cache_dir}/pretrain_clipvip_base_32.pt")
cleanDict = { key.replace("clipmodel.", "") : value for key, value in checkpoint.items() }
model =  CLIPModel(config=clipconfig)
model.load_state_dict(cleanDict)

# ------- text embedding -----
tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dir)
tokens = tokenizer(["in the forest"], padding=True, return_tensors="pt")
textOutput = model.get_text_features(**tokens)
print(textOutput.shape)

# ------- video embedding -----
processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch16", cache_dir=cache_dir)

clip_len = 12
video_frames = np.random.rand(3, 224, 224, 3)
pixel_values = processor(video_frames, return_tensors="pt").pixel_values

inputs = {
        "if_norm": True,
        "pixel_values": pixel_values}

with torch.no_grad():
    video_features = model.get_image_features(**inputs)
print(video_features.shape)

with torch.no_grad():
  sim = F.cosine_similarity(textOutput, video_features, dim=1)
  print(sim) 
  # [ 0.1142 ]