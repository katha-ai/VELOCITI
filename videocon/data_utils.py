import numpy as np
from dataloaders.control_dataset import controlDataset
from dataloaders.neg_dataset import negDataset
from dataloaders.ivat_dataset import ivatDataset
from torch.utils.data import DataLoader

def create_dataloader(dataset):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1
    )

    return dataloader

def get_data_control(args, data_dict):
    #control task
    print("\n<----- Control Task Evaluation ----->\n")
    control_dataset = controlDataset(data_dict=data_dict)
    control_dataloader = create_dataloader(dataset=control_dataset)
    return control_dataset, control_dataloader

def get_data_ivat(args, data_dict):
    print("\n<----- Caption Matching Task Evaluation ----->\n")
    vc_match_dataset = ivatDataset(data_dict=data_dict)
    vc_match_loader = create_dataloader(dataset=vc_match_dataset)
    return vc_match_dataset, vc_match_loader

def get_data_neg(args, data_dict, test):
    # negatives v2t tasks
    print("\n<----- Negative V2T Task Evaluation ----->\n")
    dataset = negDataset(data_dict=data_dict, neg_sampling=test)
    dataloader = create_dataloader(dataset=dataset)
    return dataset, dataloader

