from torch.utils.data import Dataset
from easydict import EasyDict as edict
import numpy as np
from utils.utils import get_frames_tensor, shuffle_with_constraint
import json

class negDataset(Dataset):
    """
    Make this return all possibilities of captions.
    Dataset for video-to-text tasks.
    """
    def __init__(self, data_dict, transform, neg_sampling):
        
        self.data_dict = edict(data_dict)
        self.transform = transform
        self.neg_sampling = neg_sampling
            
                
        if self.neg_sampling == 'ag_iden':
            self.ev_data = json.load(open(self.data_dict.agent_iden_caps, 'r'))
            vid_list = list(self.ev_data.keys())
        
        elif self.neg_sampling == 'ag_bind':
            self.ev_data = json.load(open(self.data_dict.agent_bind_caps, 'r'))
            vid_list = list(self.ev_data.keys())
        
        elif self.neg_sampling == 'action_bind':
            self.ev_data = json.load(open(self.data_dict.action_bind_caps, 'r'))
            vid_list = list(self.ev_data.keys())
        
        elif self.neg_sampling == 'action_mod':
            self.ev_data = json.load(open(self.data_dict.action_mod_caps, 'r'))
            vid_list = list(self.ev_data.keys())
        
        elif self.neg_sampling == 'action_adv':
            self.ev_data = json.load(open(self.data_dict.action_adv_caps, 'r'))
            vid_list = list(self.ev_data.keys())
        
        elif self.neg_sampling == 'coref':
            self.ev_data = json.load(open(self.data_dict.coref_caps, 'r'))
            vid_list = list(self.ev_data.keys())
        
        elif self.neg_sampling == 'sequence':
            self.ev_data = json.load(open(self.data_dict.seq_caps, 'r'))
            vid_list = list(self.ev_data.keys())
            

        self.vid_ev_list = []
        for vid in vid_list:
            for ev in self.ev_data[vid]:
                self.vid_ev_list.append((vid,ev))
                
    
    def __getitem__(self, idx):
        
        vid_name, ev = self.vid_ev_list[idx]
                
        frames = get_frames_tensor(frames_path=self.data_dict.frames_path,
                                   vid_name=vid_name,
                                   transform=self.transform)
        
    
        pos_cap = self.ev_data[vid_name][ev]['pos']
        neg_cap = self.ev_data[vid_name][ev]['neg']
            
        data = {
            'vid_name': vid_name,
            'event': ev,
            "frames": frames,
            "pos_cap": pos_cap,
            "neg_cap": neg_cap }
        
        return data

    def __len__(self):
        return len(self.vid_ev_list)