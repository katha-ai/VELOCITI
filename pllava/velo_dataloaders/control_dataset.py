from torch.utils.data import Dataset
from easydict import EasyDict as edict
import numpy as np
import json
from os.path import join


class controlDataset(Dataset):
    """
    Make this return all possibilities of captions.
    Dataset for video-to-text tasks.
    """

    def __init__(self, data_dict):
        """
        data_dict -> dict : containing paths of all data.
        neg_sampling -> str : 'control' or 'arg0en' or 'arg0hn' or 'verb' or 'manner' or 'event'.
        create_control -> bool : If True, create control task dataset, otherwise fetch from the pickle file.
        """

        self.data_dict = edict(data_dict)

        self.control_neg_caps = json.load(open(self.data_dict.control_neg_caps, "r"))
        self.ev_data = self.control_neg_caps
        vid_list = list(self.control_neg_caps.keys())

        self.vid_ev_list = []
        for vid in vid_list:
            for ev in self.ev_data[vid]:
                self.vid_ev_list.append((vid, ev))

    def __getitem__(self, idx):
        """
        vid_name: video name
        event: event ID
        frames: stacked tensor frames
        neg_frames: control task. Random negative video frames
        pos_cap: correct caption text
        neg_cap: control task. Random negative caption text
        """

        vid_name, ev = self.vid_ev_list[idx]

        neg_randvid_name = self.control_neg_caps[vid_name][ev]["neg_vid"]

        pos_cap = self.control_neg_caps[vid_name][ev]["pos_cap"]
        neg_cap = self.control_neg_caps[vid_name][ev]["neg_cap"]

        vid_path = join(self.data_dict.frames_path, vid_name + ".mp4")
        neg_vid_path = join(self.data_dict.frames_path, neg_randvid_name + ".mp4")

        data = {
            "vid_name": vid_name,
            "event": ev,
            "vid_path": vid_path,
            "neg_vid_path": neg_vid_path,
            "pos_cap": pos_cap,
            "neg_cap": neg_cap,
        }

        return data

    def __len__(self):
        return len(self.vid_ev_list)
