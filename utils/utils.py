from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from os.path import join
from glob import glob
import torch
from natsort import natsorted
import numpy as np
from torch.utils.data import DataLoader
import random

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
    

def convert_image_to_rgb(image):
    """
    image: PIL Image object.
    """
    return image.convert("RGB")
        
def transform(n_px=224):
    """
    Transformations, as needed for the VidSitu Dataset.
    """
    return Compose([Resize(n_px, interpolation=BICUBIC),
                    CenterCrop(n_px),
                    convert_image_to_rgb,
                    ToTensor(),
                    Normalize((0.48145466, 0.4578275, 0.40821073),
                              (0.26862954, 0.26130258, 0.27577711)),
    ])


def get_frames_tensor(frames_path, vid_name, transform=None, event_id=-1):
    
    """
    vid_name: video name of which frames are to be collected.
    event_id: event id of the video. -1 if all frames are to be collected.
    event_id sampling based on 2FPS sampling, 10s/video, 5 events. 
    """
    vid_path = join(frames_path,vid_name)

    #Possible problem with sorting, natsort preferred. Fix Needed?
    im_list = natsorted(glob(join(vid_path, '*')))
    pil_img_list = [Image.open(im) for im in im_list]

    if event_id >= 0:
        st = event_id * 4
        pil_img_list = pil_img_list[st : st+4]

    if transform is None:
        transform = transform(224)
    
    elif type(transform).__name__ == "XCLIPProcessor":
        pixel_values = transform(videos=pil_img_list, return_tensors="pt").pixel_values.squeeze()
        return pixel_values
    
    else:
        transformed_img_list = list(map(transform, pil_img_list))
        
    frames_tensor = torch.stack(transformed_img_list)
    return frames_tensor

def get_frame_embeds_tensor(frame_embeds_path, vid_name, event_id=-1, format='pt'):
    
    if format == 'pt':
        vid_path = join(frame_embeds_path,vid_name+'.pt')
        frame_embeds = torch.load(vid_path,map_location='cpu')
    elif format == 'npy':
        vid_path = join(frame_embeds_path,vid_name+'.npy')
        frame_embeds = torch.from_numpy(np.load(vid_path))
    else:
        raise ValueError('Invalid format. Choose between pt and npy feature format, or modify code accordingly.')
        
    if event_id >= 0:
        return frame_embeds[event_id*4:event_id*4+4]
    return frame_embeds

def matrix_dotprod(m1, m2):
    return torch.matmul(m1.unsqueeze(1), m2.unsqueeze(2)).squeeze(-1)

def create_dataloader(dataset, args):
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=args.pin_memory)
    
    return dataloader

def set_seed(seed_value):
    """
    Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        

import random

def shuffle_with_constraint(lst):
    """
    Shuffles the given list such that no element remains in its original position.
    e.g. [0, 1, 2, 3, 4] --> Random Shuffle --> [3, 1, 4, 0, 2] is a 'wrong' shuffle.
    Chances are unlikely, but possible --> Results in a 'fake' negative.
    """
    if len(lst) <= 1:
        return lst

    shuffled_list = lst.copy()
    original_indices = list(range(len(lst)))
    random.shuffle(original_indices)

    for i, j in enumerate(original_indices):
        while j == i:
            random.shuffle(original_indices)
            j = original_indices[i]

        shuffled_list[i] = lst[j]

    return shuffled_list