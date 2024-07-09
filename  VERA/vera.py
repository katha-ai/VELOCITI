import torch
import transformers
import os
from os.path import join
import csv
from fastprogress.fastprogress import master_bar, progress_bar
import json
import pandas as pd
from tabulate import tabulate
from glob import glob

device = torch.device("cuda:1")


def load_vera_model(use_gpu=True):
    tokenizer = transformers.AutoTokenizer.from_pretrained("liujch1998/vera")
    model = transformers.T5EncoderModel.from_pretrained("liujch1998/vera")
    model.D = model.shared.embedding_dim
    linear = torch.nn.Linear(model.D, 1, dtype=model.dtype)
    linear.weight = torch.nn.Parameter(model.shared.weight[32099, :].unsqueeze(0))
    linear.bias = torch.nn.Parameter(model.shared.weight[32098, 0].unsqueeze(0))

    if use_gpu:
        model = model.to(device)
        linear = linear.to(device)

    model.eval()
    t = model.shared.weight[32097, 0].item()
    return tokenizer, model, linear, t


def calculate_plausability_score(text, tokenizer, model, linear, t, use_gpu=True):

    input_ids = tokenizer(
        text, return_tensors="pt", padding="longest", truncation=True, max_length=512
    ).input_ids
    input_ids = input_ids.to(device)

    with torch.no_grad():
        output = model(input_ids)
        last_hidden_state = output.last_hidden_state
        hidden = last_hidden_state[0, -1, :]
        logit = linear(hidden).squeeze(-1)
        logit_calibrated = logit / t
        score_calibrated = logit_calibrated.sigmoid()

    return score_calibrated.item()


data_root = r"data2/"
outdir = r"output2/"


data_dict = {
    "vidsitu_dict_path": f"{data_root}/vidsitu_dict.json",
    "agent_iden_caps": f"{data_root}/agent_iden.json",
    "agent_bind_caps": f"{data_root}/agent_bind.json",
    "action_bind_caps": f"{data_root}/action_bind.json",
    "action_mod_caps": f"{data_root}/action_mod.json",
    "control_neg_caps": f"{data_root}/control.json",
    "coref_caps": f"{data_root}/coref.json",
    "seq_caps": f"{data_root}/sequence.json",
    "action_adv_caps": f"{data_root}/action_adv.json",
}


tokenizer, model, linear, t = load_vera_model()

mb = master_bar(data_dict.items())

for key, value in mb:
    print(f"Processing {key}...")
    log_outdir = join(outdir, key + ".csv")
    os.makedirs(outdir, exist_ok=True)

    header = [
        "video",
        "event",
        "pos_cap",
        "neg_cap",
        "pos_plaus_score",
        "neg_plaus_score",
    ]
    data = json.load(open(value, "r"))

    if key in [
        "agent_iden_caps",
        "agent_bind_caps",
        "action_bind_caps",
        "action_mod_caps",
        "action_adv_caps",
        "coref_caps",
        "seq_caps",
    ]:

        with open(log_outdir, "w") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(header)
            for vid in progress_bar(data):
                for ev in data[vid]:
                    pos_cap = data[vid][ev]["pos"]
                    neg_cap = data[vid][ev]["neg"]
                    pos_plaus_score = calculate_plausability_score(
                        pos_cap, tokenizer, model, linear, t
                    )
                    neg_plaus_score = calculate_plausability_score(
                        neg_cap, tokenizer, model, linear, t
                    )
                    log_text = [
                        vid,
                        ev,
                        pos_cap,
                        neg_cap,
                        pos_plaus_score,
                        neg_plaus_score,
                    ]

                csvwriter.writerow(log_text)

    elif key in ["control_neg_caps"]:

        with open(log_outdir, "w") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(header)

            for vid in progress_bar(data):
                for event in data[vid]:
                    pos_cap = data[vid][event]["pos_cap"]
                    neg_cap = data[vid][event]["neg_cap"]

                    pos_plaus_score = calculate_plausability_score(
                        pos_cap, tokenizer, model, linear, t
                    )
                    neg_plaus_score = calculate_plausability_score(
                        neg_cap, tokenizer, model, linear, t
                    )
                    log_text = [
                        vid,
                        event,
                        pos_cap,
                        neg_cap,
                        pos_plaus_score,
                        neg_plaus_score,
                    ]
                    csvwriter.writerow(log_text)


metrics = {}
for file in glob(data_root + "*.csv"):
    task = os.path.basename(file).replace("_caps.csv", "")
    df = pd.read_csv(file)
    df["correct"] = df["pos_plaus_score"] > df["neg_plaus_score"]
    correct = df["correct"].sum()
    total = len(df)
    metrics[task] = correct / total


print(tabulate(metrics.items(), headers=["Task", "Accuracy"], tablefmt="grid"))