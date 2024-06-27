import numpy as np
import av
from dataloaders.control_dataset import controlDataset
from dataloaders.neg_dataset import negDataset
from dataloaders.ivat_dataset import ivatDataset
from utils.utils import create_dataloader, set_seed


def get_data_control(args, data_dict):
    #control task
    print("\n<----- Control Task Evaluation ----->\n")
    control_dataset = controlDataset(data_dict=data_dict,frames_flag=False)
    control_dataloader = create_dataloader(dataset=control_dataset, args=args)
    return control_dataset, control_dataloader

def get_data_ivat(args, data_dict):
    print("\n<----- Caption Matching Task Evaluation ----->\n")
    vc_match_dataset = ivatDataset(data_dict=data_dict, frames_flag=False)
    vc_match_loader = create_dataloader(dataset=vc_match_dataset, args=args)
    return vc_match_dataset, vc_match_loader

def get_data_neg(args, data_dict, test):
    # negatives v2t tasks
    dataset = negDataset(data_dict=data_dict, frames_flag=False, neg_sampling=test)
    dataloader = create_dataloader(dataset=dataset, args=args)
    return dataset, dataloader

def read_video_pyav(video_path):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    container = av.open(video_path)

    # sample uniformly 8 frames from the video
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
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