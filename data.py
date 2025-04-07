from datasets import load_dataset

from torch.utils.data import Dataset, DataLoader
from easydict import EasyDict as edict
import numpy as np
from utils.clip_utils import get_frames_tensor
import json


class negDataset(Dataset):
    """
    Make this return all possibilities of captions.
    Dataset for video-to-text tasks.
    """

    def __init__(self, data_dict, frames_path, transform=None ):

        self.data_dict = data_dict
        self.frames_path = frames_path
        self.transform = transform


    def __getitem__(self, idx):

        test_name = self.data_dict[idx]['test_name']
        video_id, ev = self.data_dict[idx]['video_id'].split('.')[0], self.data_dict[idx]['event']
        pos_cap, neg_cap = self.data_dict[idx]['pos'], self.data_dict[idx]['neg']

        if self.transform:
            frames = get_frames_tensor(
                frames_path=self.frames_path,
                vid_name=video_id,
                transform=self.transform,
            )
        else:
            frames = "{}/{}.mp4".format(self.frames_path, video_id)


        data = {
            "test_name": test_name,
            "video_id": video_id,
            "ev": ev,
            "frames": frames,
            "pos": pos_cap,
            "neg": neg_cap,
        }

        return data

    def __len__(self):
        return len(self.data_dict)
    


def get_data(args, test_name=None, transform=None):
    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("katha-ai-iiith/VELOCITI", cache_dir=args.data_root)
    dataset = ds['test']

    if test_name:
        dataset = dataset.filter(lambda example: example['test_name'] == test_name)

    dataset = negDataset(dataset, args.frames_root, transform=transform)
    dataloder = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=args.num_workers, pin_memory=args.pin_memory)

    return dataset, dataloder