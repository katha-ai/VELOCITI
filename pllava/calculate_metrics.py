import pandas as pd
import os
from glob import glob
from natsort import natsorted
import numpy as np
from tabulate import tabulate
import argparse
import torch

csv_files = natsorted(glob("RESULTS/*.csv"))
metrics = {}


def wino(va_ca_score, va_cb_score, vb_ca_score, vb_cb_score):
    gt_res = torch.logical_and((va_ca_score > vb_ca_score), (vb_cb_score > va_cb_score))
    gv_res = torch.logical_and((va_ca_score > va_cb_score), (vb_cb_score > vb_ca_score))
    group_res = torch.logical_and(gt_res, gv_res)

    gt_cnt = gt_res.sum()
    gv_cnt = gv_res.sum()
    group_cnt = group_res.sum()

    ind_gt_cnt = (va_ca_score > vb_ca_score).sum() + (vb_cb_score > va_cb_score).sum()
    ind_gv_cnt = (va_ca_score > va_cb_score).sum() + (vb_cb_score > vb_ca_score).sum()
    return torch.tensor([gt_cnt, gv_cnt, group_cnt, ind_gt_cnt, ind_gv_cnt])


for file in csv_files:
    df = pd.read_csv(file)
    neg = os.path.basename(file).replace(".csv", "")

    if neg in [
        "ag_iden",
        "ag_bind",
        "action_bind",
        "action_mod",
        "action_adv",
        "coref",
        "sequence",
        "control_t2v",
        "control_v2t",
    ]:
        df["correct"] = df["entail_score_pos"] > df["entail_score_neg"]
        # sum of correct predictions
        correct = df["correct"].sum()
        acc = correct / len(df)
        metrics[neg] = acc

    elif neg == "ivat":
        v1c1 = torch.tensor(df["entail_score_v1c1"])
        v1c2 = torch.tensor(df["entail_score_v1c2"])
        v2c1 = torch.tensor(df["entail_score_v2c1"])
        v2c2 = torch.tensor(df["entail_score_v2c2"])

        results = wino(v1c1, v1c2, v2c1, v2c2)

        metrics["t2v"] = results[0].item() / len(df)
        metrics["v2t"] = results[1].item() / len(df)
        metrics["group"] = results[2].item() / len(df)

        metrics["ind_t2v"] = results[3].item() / (2 * len(df))
        metrics["ind_v2t"] = results[4].item() / (2 * len(df))

df = pd.DataFrame([metrics])
print(tabulate(df, headers="keys", tablefmt="psql"))
