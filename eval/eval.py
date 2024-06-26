import torch
from transformers import AutoProcessor

def get_video_features(model, frames, device, model_type='open-clip'):
    
    if len(frames.shape) == 3:
        b,f,dim = frames.shape
        image_features = frames.to(device) #b,20,512    
        return image_features
    
    else:
        
        b,f,c,h,w = frames.shape
        images = frames.reshape((b*f,c,h,w))
        images = images.to(device)
        
        if model_type == "open_clip" or model_type=="neg_clip" or model_type=="vificlip":
            image_features = model.encode_image(images, normalize=False)
            
            image_features = image_features.reshape((b,f,-1)).to(device) #b,20,512
            
            image_features = image_features.mean(dim=1) #b,512
            image_features /= image_features.norm(dim=-1, keepdim=True) #b,512
            
            return image_features
        
        elif model_type == "clip_vip":
            # TODO: The frames incoming from get_frames_tensor would likely throw an error here for [0,1] norm?
            # TODO: need to either fix that or norm here? -> Not sure.
            # TODO: Check HF Home Environ Globally Set? [Shouldn't download cache in default dir]
            
            #processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch16")
            #pixel_values = processor(videos=list(images), return_tensors="pt").pixel_values
            frames = frames.to(device)
            inputs = {"if_norm": True,
                      "pixel_values": frames}
            
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            
            return image_features
            
            
        elif model_type == "vificlip":
            pass


def get_cap_features(model, cap, tokenizer, device, model_type='open_clip'):
    
    
    with torch.no_grad():
        
        if model_type == "open_clip" or model_type=="vificlip":
            text = tokenizer(cap)
            text = text.to(device)
            
            text_features = model.encode_text(text, normalize=True)
        
        elif model_type == "clip_vip":
            text = tokenizer(cap, padding=True, return_tensors="pt", truncation=True)
            text = text.to(device)
            
            text_features = model.get_text_features(**text, if_norm=True)
        
        elif model_type == "neg_clip":
            text = tokenizer(cap)
            text = text.to(device)
            text_features = model.encode_text(text, normalize=True)
            
        #TODO: Check if the feats need to be normed for all models?
        
        # text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.to(device)
    
    return text_features
