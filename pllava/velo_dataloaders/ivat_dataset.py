from torch.utils.data import Dataset
from easydict import EasyDict as edict
import numpy as np
import torch
import json
from os.path import join


class ivatDataset(Dataset):
    """
    Dataset class for event-caption-matching task. For each video, EV1 and EV5 frames and captions are returned.
    """

    def __init__(self, data_dict):
        """
        data_dict -> dict : containing paths of all data.
        """
        self.data_dict = edict(data_dict)

        self.vidsitu_dict = json.load(open(self.data_dict.vidsitu_dict_path, "r"))

        self.vid_list = list(self.vidsitu_dict.keys())

    def __getitem__(self, idx):
        vid_name = self.vid_list[idx]
        pos_cap_ev1 = self.vidsitu_dict[vid_name]["Ev1"]["pos"]
        pos_cap_ev5 = self.vidsitu_dict[vid_name]["Ev5"]["pos"]

        vid_path_ev1 = join(self.data_dict.vid_path_4s, vid_name, vid_name + "_p1.mp4")
        vid_path_ev5 = join(self.data_dict.vid_path_4s, vid_name, vid_name + "_p2.mp4")

        return {
            "vid_name": vid_name,
            "event": "(Ev1, Ev5)",
            "vid_path_ev1": vid_path_ev1,
            "vid_path_ev5": vid_path_ev5,
            "pos_cap_ev1": pos_cap_ev1,
            "pos_cap_ev5": pos_cap_ev5,
        }

    def __len__(self):
        return len(self.vid_list)
