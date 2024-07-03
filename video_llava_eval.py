import os
import argparse
from utils.utils import set_seed
from video_llava.eval import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="video_llava",
        help="Model architecture to be used",
        choices=["video_llava"],
    )
    parser.add_argument(
        "--test",
        type=str,
        default="ag_iden",
        help="test name",
        choices=[
            "ivat",
            "control_t2v",
            "control_v2t",
            "ag_iden",
            "ag_bind",
            "action_bind",
            "action_mod",
            "action_adv",
            "coref",
            "sequence",
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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=1000)

    args = parser.parse_args()

    data_dict = {
        "vidsitu_dict_path": f"{args.data_root}/vidsitu_dict.json",
        "frames_path": f"{args.data_root}/frames",
        "videos_10s_path": f"{args.data_root}/videos/velociti_videos_10s",
        "videos_4s_path": f"{args.data_root}/videos/velociti_videos_4s",
        "agent_iden_caps": f"{args.data_root}/agent_iden.json",
        "agent_bind_caps": f"{args.data_root}/agent_bind.json",
        "action_bind_caps": f"{args.data_root}/action_bind.json",
        "action_mod_caps": f"{args.data_root}/action_mod.json",
        "control_neg_caps": f"{args.data_root}/control.json",
        "coref_caps": f"{args.data_root}/coref.json",
        "seq_caps": f"{args.data_root}/sequence.json",
        "action_adv_caps": f"{args.data_root}/action_adv.json",
    }

    os.makedirs(args.output, exist_ok=True)
    set_seed(args.seed)

    if args.test == "ivat":
        ivat(args, data_dict)
    elif args.test == "control_t2v":
        text2video(args, data_dict)
    else:
        video2text(args, data_dict)
