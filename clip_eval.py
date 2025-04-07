import os
from fastprogress.fastprogress import progress_bar
import pandas as pd
import argparse

import torch
from transformers import AutoProcessor

from utils.clip_utils import get_frames_tensor, matrix_dotprod
import configs.model_card as models
from models.clip_model import load_clip_model, get_cap_features, get_video_features
from data import get_data



def clip_eval(model, model_type, tokenizer, dataloader, device):

    df = pd.DataFrame(
            columns=[
                "test_name",
                "video_id",
                "event",
                "pos_score",
                "neg_score",
            ]
        )
    cnt, tot = 0, 0

    for item in progress_bar(dataloader):
        test_name = item['test_name']
        frames = item['frames']
        video_id, ev = item['video_id'], item['ev']
        pos_cap, neg_cap = item['pos'], item['neg']

        video_features = get_video_features(model, frames, device, model_type=model_type)
        pos_cap_features = get_cap_features(model, pos_cap, tokenizer, device, model_type=model_type)
        neg_cap_features = get_cap_features(model, neg_cap, tokenizer, device, model_type=model_type)

        pos_score = matrix_dotprod(video_features, pos_cap_features)
        neg_score = matrix_dotprod(video_features, neg_cap_features)

        cnt += (pos_score > neg_score).sum().item()
        tot += frames.shape[0]

        tmp_df = pd.DataFrame(
                    {
                        "test_name": item["test_name"],
                        "video_id": video_id,
                        "event": item["ev"],
                        "pos_score": pos_score.reshape(-1).detach().cpu().tolist(),
                        "neg_score": neg_score.reshape(-1).detach().cpu().tolist(),
                    }
                )
        df = pd.concat([df, tmp_df])
        df.to_csv('temp.csv', index=False)

    return (cnt / tot), df


def main(args):

    device = torch.device(args.device)
    
    if args.all:
        models_list = models.keys()
    else:
        models_list = [args.model]

    for model in models_list:
        args.model = model
        model, tokenizer, transform, model_type = load_clip_model(args, device)

        tests = {'action_adv', 'action_bind', 'action_manner', 'agent_bind', 'agent_random', 'chrono', 'control', 'coref'}

        for test_name in tests:
            dataset, dataloader = get_data(args, test_name, transform=transform)
            acc, df = clip_eval(model, model_type, tokenizer, dataloader, device)

            if args.exhaustive_log:
                out_dir = f'{args.output}/{args.model}'
                os.makedirs(args.output, exist_ok=True)

                filename = f'{out_dir}/{test_name}.csv'
                df.to_csv(filename, index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="clip_B_32",
        help="Model architecture to be used",
        choices=[
            "clip_B_32",
            "clip_L_14",
            "evaclip_L_14",
            "siglip_B_16",
            "siglip_L_16",
            "negclip_B_32",
            "vificlip",
        ],
    )
    parser.add_argument(
        "--cache_dir",
        default=".hfcache",
        type=str,
        help="Directory to where downloaded models are cached",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/",
        help="Directory to where results are saved",
    )
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--frames_root", type=str, default="./frames")
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Whether to test all the pretrained models in the paper",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--exhaustive_log",
        action="store_true",
        default=True,
        help="wether to log the results of each sample",
    )

    args = parser.parse_args()
    main(args)