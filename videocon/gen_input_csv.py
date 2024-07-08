import os
import argparse
import csv
from tqdm import tqdm

from data_utils import *

PROMPT = '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <|video|>
Human: Carefully watch the video and pay attention to the sequence of events, the details and actions of persons. Based on your observation, does the given video entail the caption?
Caption: {}
AI: '''

def ivat(args, data_dict):

    dataset, dataloader = get_data_ivat(args, data_dict)

    fields = ['videopath','text','caption','video_id','item']
    inp_filename = "{}/{}.csv".format(args.output_folder,args.test)
 
    with open(inp_filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fields)

        for i, (data_batch) in enumerate(tqdm(dataloader)):
            vid_name, event = data_batch['vid_name'][0], data_batch['event'][0]
            video_a, video_b = data_batch['vid_path_ev1'][0], data_batch['vid_path_ev5'][0]
            cap_a_text, cap_b_text = data_batch['pos_cap_ev1'][0], data_batch['pos_cap_ev5'][0]

            cap_a,cap_b = PROMPT.format(cap_a_text),PROMPT.format(cap_b_text)

            writer.writerow([video_a,cap_a_text,cap_a,vid_name,'va_ca'])
            writer.writerow([video_a,cap_b_text,cap_b,vid_name,'vb_cb'])
            writer.writerow([video_b,cap_a_text,cap_a,vid_name,'vb_ca'])
            writer.writerow([video_b,cap_b_text,cap_b,vid_name,'vb_cb'])



def video2text(args, data_dict):
    print(args.test)

    if args.test == 'control_v2t':
        dataset, dataloader = get_data_control(args, data_dict)
    else:
        dataset, dataloader = get_data_neg(args, data_dict, args.test)

    fields = ['videopath','text','caption','video_id','ev','item']
    inp_filename = "{}/{}.csv".format(args.output_folder,args.test)
 
    with open(inp_filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fields)

        for i, (data_batch) in enumerate(tqdm(dataloader)):
            vid_name, ev = data_batch['vid_name'][0], data_batch['event'][0]
            video_path = data_batch['vid_path'][0]
            pos_text, neg_text = data_batch['pos_cap'][0], data_batch['neg_cap'][0]

            pos_cap, neg_cap = PROMPT.format(pos_text), PROMPT.format(neg_text)

            writer.writerow([video_path,pos_text,pos_cap,vid_name,ev,'pos'])
            writer.writerow([video_path,neg_text,neg_cap,vid_name,ev,'neg'])

    

def text2video(args, data_dict):
    dataset, dataloader = get_data_control(args, data_dict)

    fields = ['videopath','text','caption','video_id','ev','item']
    inp_filename = "{}/{}.csv".format(args.output_folder,args.test)
 
    with open(inp_filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fields)

        for i, (data_batch) in enumerate(tqdm(dataloader)):
            vid_name, ev = data_batch['vid_name'][0], data_batch['event'][0]
            pos_video_path, neg_video_path = data_batch['vid_path'][0], data_batch['neg_vid_path'][0]
            text = data_batch['pos_cap'][0]

            cap = PROMPT.format(text)
            
            writer.writerow([pos_video_path,text,cap,vid_name,ev,'pos'])
            writer.writerow([neg_video_path,text,cap,vid_name,ev,'neg'])

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
        "--output_folder",
        type=str,
        default="videocon_prompts/",
        help="Directory to where outputs are saved",
    )
    parser.add_argument("--data_root", type=str, default="./data")


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

    os.makedirs(args.output_folder, exist_ok=True)

    if args.test == "ivat":
        ivat(args, data_dict)
    elif args.test == "control_t2v":
        text2video(args, data_dict)
    else:
        video2text(args, data_dict)