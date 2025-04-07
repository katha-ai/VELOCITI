from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from os.path import join
from glob import glob
import torch
from natsort import natsorted
import numpy as np
from torch.utils.data import DataLoader
import random
import av


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
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def get_frames_tensor(frames_path, vid_name, transform=None, event_id=-1):
    """
    vid_name: video name of which frames are to be collected.
    event_id: event id of the video. -1 if all frames are to be collected.
    event_id sampling based on 2FPS sampling, 10s/video, 5 events.
    """
    vid_path = join(frames_path, vid_name)

    # Possible problem with sorting, natsort preferred. Fix Needed?
    im_list = natsorted(glob(join(vid_path, "*")))
    pil_img_list = [Image.open(im) for im in im_list]

    if event_id >= 0:
        st = event_id * 4
        pil_img_list = pil_img_list[st : st + 4]

    if transform is None:
        transform = transform(224)

    elif type(transform).__name__ == "XCLIPProcessor":
        pixel_values = transform(
            videos=pil_img_list, return_tensors="pt"
        ).pixel_values.squeeze()
        return pixel_values

    else:
        transformed_img_list = list(map(transform, pil_img_list))

    frames_tensor = torch.stack(transformed_img_list)
    return frames_tensor


def matrix_dotprod(m1, m2):
    return torch.matmul(m1.unsqueeze(1), m2.unsqueeze(2)).squeeze(-1)



def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


    
def get_frame_indices(fps, total_frames):
    # Calculate the duration of the video in seconds
    duration = total_frames / fps

    # Get the middle of each second (0.5s, 1.5s, 2.5s, etc.)
    middle_seconds = np.arange(0.5, duration, 1)

    # Convert these times into frame indices
    frame_indices = (middle_seconds * fps).astype(int)
    
    return frame_indices


def process_video(video_path):
    container = av.open(video_path)

    total_frames = container.streams.video[0].frames
    fps = round(container.streams.video[0].base_rate.numerator/container.streams.video[0].base_rate.denominator)
    indices = get_frame_indices(fps, total_frames)
    
    clip = read_video_pyav(container, indices)

    return clip