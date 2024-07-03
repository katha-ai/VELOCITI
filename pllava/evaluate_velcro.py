from easydict import EasyDict as edict
from tasks.eval.model_utils import load_pllava, pllava_answer
import torch
import torchvision
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
from tasks.eval.eval_utils import conv_templates
from torch.utils.data import DataLoader
from velo_dataloaders.neg_dataset import negDataset
from velo_dataloaders.control_dataset import controlDataset
from velo_dataloaders.ivat_dataset import ivatDataset
from fastprogress.fastprogress import master_bar, progress_bar
import csv
import os
from os.path import join
import argparse
import random


def set_seed(seed_value):
    """
    Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def load_model(args, device):
    model, processor = load_pllava(
        repo_id=args.pretrained_model_name_or_path,
        num_frames=args.num_frames,
        use_lora=args.use_lora,
        lora_alpha=args.lora_alpha,
        weight_dir=args.weight_dir,
        pooling_shape=(16, 12, 12),
    )

    model = model.to(device)
    model = model.eval()

    return model, processor


def single_test(
    model,
    processor,
    vid_path,
    query="Describe the video in details.",
    num_frames=4,
    conv_mode="plain",
    cal_entail=False,
):
    def get_index(num_frames, num_segments):
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
        offsets = np.array(
            [start + int(np.round(seg_size * idx)) for idx in range(num_segments)]
        )
        return offsets

    def load_video(
        video_path, num_segments=8, return_msg=False, num_frames=4, resolution=336
    ):
        transforms = torchvision.transforms.Resize(size=resolution)
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        num_frames = len(vr)
        frame_indices = get_index(num_frames, num_segments)
        images_group = list()
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(transforms(img))
        if return_msg:
            fps = float(vr.get_avg_fps())
            sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
            # " " should be added in the start and end
            msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
            return images_group, msg
        else:
            return images_group

    if num_frames != 0:
        vid, msg = load_video(
            vid_path, num_segments=num_frames, return_msg=True, resolution=336
        )
    else:
        vid, msg = None, "num_frames is 0, not inputing image"
    img_list = vid

    conv = conv_templates[conv_mode].copy()
    conv.user_query(query, is_mm=True)

    llm_response, entailment_score = pllava_answer(
        conv=conv,
        model=model,
        processor=processor,
        do_sample=False,
        img_list=img_list,
        max_new_tokens=1,
        print_res=False,
        temperature=0.0,
        cal_entail=cal_entail,
    )

    if entailment_score is None:
        return llm_response
    else:
        return llm_response, entailment_score


parser = argparse.ArgumentParser()
parser.add_argument(
    "--pretrained_model_name_or_path", type=str, default="MODELS/pllava-7b"
)
parser.add_argument("--num_frames", type=int, default=4)
parser.add_argument("--use_lora", action="store_true", default=True)
parser.add_argument("--use_multi_gpus", action="store_true", default=False)
parser.add_argument("--weight_dir", type=str, default="MODELS/pllava-7b")
parser.add_argument("--conv_mode", type=str, default="plain")
parser.add_argument("--lora_alpha", type=int, default=4)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--videos_dir", type=str, default="data/videos/velociti_videos_10s")
parser.add_argument(
    "--videos_dir_4s", type=str, default="data/videos/velociti_videos_4s"
)
parser.add_argument("--outdir", type=str, default="RESULTS")
parser.add_argument("--data_root", type=str, default="data")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=1000)

args = parser.parse_args()

set_seed(args.seed)
device = torch.device(args.device)

model, processor = load_model(args=args, device=device)
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
base_query = "Carefully watch the video and pay attention to the sequence of events, the details and actions of persons. Based on your observation, does the given video entail the caption?"

data_dict = {
    "vidsitu_dict_path": f"data/vidsitu_dict.json",
    "frames_path": f"/workspace/darshan/velcro_vids",
    "vid_path_4s": f"/workspace/darshan/val_vids_split4s_960/",
    "agent_iden_caps": f"data/agent_iden.json",
    "agent_bind_caps": f"data/agent_bind.json",
    "action_bind_caps": f"data/action_bind.json",
    "action_mod_caps": f"data/action_mod.json",
    "control_neg_caps": f"data/control.json",
    "coref_caps": f"data/coref.json",
    "seq_caps": f"data/sequence.json",
    "action_adv_caps": f"data/action_adv.json",
}


def perform_neg_inference(neg_type):
    headers = [
        "video",
        "event",
        "pos_cap",
        "neg_cap",
        "pllava_response_pos",
        "pllava_response_neg",
        "entail_score_pos",
        "entail_score_neg",
    ]

    print(f"<<<<<<<<< --------- Performing for {neg_type} --------->>>>>>>>>>")
    csv_file = neg_type + ".csv"
    csv_file = join(args.outdir, csv_file)

    if not os.path.isfile(csv_file):
        with open(csv_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

    dataset = negDataset(data_dict=data_dict, neg_sampling=neg_type)

    dataloader = DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
    )

    all_samples = progress_bar(dataloader)
    with open(csv_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for item in all_samples:
            pos_query = base_query + f" Caption: {item['pos_cap'][0]}"
            neg_query = base_query + f" Caption: {item['neg_cap'][0]}"
            # 1 video, 2 captions setup
            pllava_response_pos, entail_score_pos = single_test(
                model=model,
                processor=processor,
                vid_path=item["vid_path"][0],
                query=pos_query,
                cal_entail=True,
            )
            pllava_response_neg, entail_score_neg = single_test(
                model=model,
                processor=processor,
                vid_path=item["vid_path"][0],
                query=neg_query,
                cal_entail=True,
            )
            event_details = item["event"][0]
            data = [
                item["vid_name"][0],
                event_details,
                item["pos_cap"][0],
                item["neg_cap"][0],
                pllava_response_pos,
                pllava_response_neg,
                entail_score_pos,
                entail_score_neg,
            ]

            writer.writerow(data)


def perform_control_eval(control_type):
    """
    control_type -- takes value, either 't2v' or 'v2t'.
    Return: return_description
    """

    print(
        f"<<<<<<<<< --------- Performing for Control-{control_type} --------->>>>>>>>>>"
    )

    csv_file = "control_" + control_type + ".csv"
    csv_file = join(args.outdir, csv_file)

    dataset = controlDataset(data_dict=data_dict)
    dataloader = DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
    )

    all_samples = progress_bar(dataloader)

    if control_type == "t2v":
        # one text against 2 videos, one correct and the other incorrect.
        headers = [
            "video",
            "event",
            "pos_cap",
            "neg_vid",
            "pllava2_response_pos",
            "pllava2_response_neg",
            "entail_score_pos",
            "entail_score_neg",
        ]
        if not os.path.isfile(csv_file):
            with open(csv_file, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
        with open(csv_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for item in all_samples:
                pos_query = base_query + f" Caption: {item['pos_cap'][0]}"
                pllava_response_pos, entail_score_pos = single_test(
                    model=model,
                    processor=processor,
                    vid_path=item["vid_path"][0],
                    query=pos_query,
                    cal_entail=True,
                )
                pllava_response_neg, entail_score_neg = single_test(
                    model=model,
                    processor=processor,
                    vid_path=item["neg_vid_path"][0],
                    query=pos_query,
                    cal_entail=True,
                )
                event_details = item["event"][0]
                neg_vid_name = os.path.basename(item["neg_vid_path"][0]).replace(
                    ".mp4", ""
                )
                data = [
                    item["vid_name"][0],
                    event_details,
                    item["pos_cap"][0],
                    neg_vid_name,
                    pllava_response_pos,
                    pllava_response_neg,
                    entail_score_pos,
                    entail_score_neg,
                ]
                writer.writerow(data)

    elif control_type == "v2t":
        # one video against 2 text, one correct and the other incorrect.
        headers = [
            "video",
            "event",
            "pos_cap",
            "neg_cap",
            "pllava2_response_pos",
            "pllava2_response_neg",
            "entail_score_pos",
            "entail_score_neg",
        ]
        if not os.path.isfile(csv_file):
            with open(csv_file, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
        with open(csv_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for item in all_samples:
                pos_query = base_query + f" Caption: {item['pos_cap'][0]}"
                neg_query = base_query + f" Caption: {item['neg_cap'][0]}"
                pllava_response_pos, entail_score_pos = single_test(
                    model=model,
                    processor=processor,
                    vid_path=item["vid_path"][0],
                    query=pos_query,
                    cal_entail=True,
                )
                pllava_response_neg, entail_score_neg = single_test(
                    model=model,
                    processor=processor,
                    vid_path=item["vid_path"][0],
                    query=neg_query,
                    cal_entail=True,
                )
                event_details = item["event"][0]
                data = [
                    item["vid_name"][0],
                    event_details,
                    item["pos_cap"][0],
                    item["neg_cap"][0],
                    pllava_response_pos,
                    pllava_response_neg,
                    entail_score_pos,
                    entail_score_neg,
                ]
                writer.writerow(data)


def perform_match():
    headers = [
        "video_1",
        "video_2",
        "pos_1_cap",
        "pos_2_cap",
        "entail_score_v1c1",
        "entail_score_v1c2",
        "entail_score_v2c1",
        "entail_score_v2c2",
    ]

    print(f"<<<<<<<<< --------- Performing IVAT --------->>>>>>>>>>")
    csv_file = "ivat.csv"
    csv_file = join(args.outdir, csv_file)

    if not os.path.isfile(csv_file):
        with open(csv_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

    dataset = ivatDataset(data_dict=data_dict)

    dataloader = DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
    )

    all_samples = progress_bar(dataloader)
    with open(csv_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for item in all_samples:
            vid_ev1 = item["vid_path_ev1"][0]
            vid_ev5 = item["vid_path_ev5"][0]

            cap_ev1 = base_query + f" Caption: {item['pos_cap_ev1'][0]}"
            cap_ev5 = base_query + f" Caption: {item['pos_cap_ev5'][0]}"

            _, entail_score_v1c1 = single_test(
                model=model,
                processor=processor,
                vid_path=vid_ev1,
                query=cap_ev1,
                cal_entail=True,
            )
            _, entail_score_v1c2 = single_test(
                model=model,
                processor=processor,
                vid_path=vid_ev1,
                query=cap_ev5,
                cal_entail=True,
            )

            _, entail_score_v2c1 = single_test(
                model=model,
                processor=processor,
                vid_path=vid_ev5,
                query=cap_ev1,
                cal_entail=True,
            )
            _, entail_score_v2c2 = single_test(
                model=model,
                processor=processor,
                vid_path=vid_ev5,
                query=cap_ev5,
                cal_entail=True,
            )

            vid_1 = os.path.basename(vid_ev1).replace(".mp4", "")
            vid_2 = os.path.basename(vid_ev5).replace(".mp4", "")
            data = [
                vid_1,
                vid_2,
                cap_ev1,
                cap_ev5,
                entail_score_v1c1,
                entail_score_v1c2,
                entail_score_v2c1,
                entail_score_v2c2,
            ]

            writer.writerow(data)


perform_neg_inference(neg_type="ag_iden")
perform_neg_inference(neg_type="ag_bind")
perform_neg_inference(neg_type="action_bind")
perform_neg_inference(neg_type="action_mod")
perform_neg_inference(neg_type="action_adv")
perform_neg_inference(neg_type="coref")
perform_neg_inference(neg_type="sequence")

perform_control_eval(control_type="t2v")
perform_control_eval(control_type="v2t")

perform_match()
