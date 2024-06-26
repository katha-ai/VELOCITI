from torch.utils.data import Dataset
from easydict import EasyDict as edict
import numpy as np
from utils.utils import get_frames_tensor
import torch
import json

class ivatDataset(Dataset):
    """
    Dataset class for event-caption-matching task. For each video, EV1 and EV5 frames and captions are returned.
    """
    def __init__(self, data_dict, transform):
        """
        data_dict -> dict : containing paths of all data.
        """
        self.data_dict = edict(data_dict)
        self.transform = transform
        
        self.vidsitu_dict = json.load(open(self.data_dict.vidsitu_dict_path, 'r'))
        
        self.vid_list = list(self.vidsitu_dict.keys())
                
    def __getitem__(self, idx):
        vid_name = self.vid_list[idx]
        
        frames_ev1 = torch.cat((get_frames_tensor(frames_path=self.data_dict.frames_path,
                                                  vid_name=vid_name,
                                                  event_id=0,
                                                  transform=self.transform),
                                
                                get_frames_tensor(frames_path=self.data_dict.frames_path,
                                                  vid_name=vid_name,
                                                  event_id=1,
                                                  transform=self.transform)))
        
        frames_ev5 = torch.cat((get_frames_tensor(frames_path=self.data_dict.frames_path,
                                                  vid_name=vid_name,
                                                  event_id=3,
                                                  transform=self.transform),
                                
                                get_frames_tensor(frames_path=self.data_dict.frames_path,
                                                  vid_name=vid_name,
                                                  event_id=4,
                                                  transform=self.transform)))
        
        
        pos_cap_ev1 = self.vidsitu_dict[vid_name]['Ev1']['pos']
        pos_cap_ev5 = self.vidsitu_dict[vid_name]['Ev5']['pos']
        
        
        return {
            'vid_name': vid_name,
            'event': '(Ev1, Ev5)',
            "frames_ev1": frames_ev1,
            "frames_ev5": frames_ev5,
            "pos_cap_ev1": pos_cap_ev1,
            "pos_cap_ev5": pos_cap_ev5
        }

    def __len__(self):
        return len(self.vid_list)
    