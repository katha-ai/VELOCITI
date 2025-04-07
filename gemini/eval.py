import time
import argparse
import json
import google.generativeai as genai
import pandas as pd
from fastprogress.fastprogress import progress_bar

import os
from os.path import join
from tqdm import tqdm
import random
random.seed(1024)
from tabulate import tabulate
from collections import OrderedDict

from data import get_data

# Global dictionary to track uploaded videos
uploaded_vids = {}

def upload_to_gcp(dataset, args):
    global uploaded_vids  # Use global to maintain state across tests
    vid_gcp = {}
    print("Started upload to GCP")
    
    for item in dataset:
        vid_name = item['video_id']
        # Skip upload if video has already been uploaded
        if vid_name in uploaded_vids:
            vid_gcp[vid_name] = uploaded_vids[vid_name]  # Reuse the uploaded reference
            continue
        
        vid_path = join(args.video_root, vid_name + '.mp4')
        # print(vid_path)
        
        if not os.path.exists(vid_path):
            print("Video not found: ", vid_path)
            continue
        
        vid_gcp[vid_name] = genai.upload_file(path=vid_path)
        uploaded_vids[vid_name] = vid_gcp[vid_name]  # Store uploaded video reference in global dict

    print("Completed upload to GCP")
    return vid_gcp



def delete_from_gcp():
    print("Deleting videos from GCP")
    for k, v in tqdm(uploaded_vids.items(), total=len(uploaded_vids)):
        genai.delete_file(v.name)
    print("Deleting done")



def create_model(model='gemini-1.5-flash'):
    generation_config = {
    "temperature": 0.,
    "max_output_tokens": 300
    }
    
    safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    },
    ]
    
    model = genai.GenerativeModel(model_name=model,
                                generation_config=generation_config,
                                safety_settings=safety_settings)

    return model


def get_prompt(type='entail'):

    entail_prompt = "Carefully watch the video and pay attention to the sequence of events, the details and actions of persons.\nHere is a caption that describes the video: {cap}\nBased on your observation, does the given video entail the caption? Just answer with either Yes or No."

    mcq_prompt = "Carefully watch the video and pay attention to the sequence of events, the details and actions of persons. Here are two captions that describe the video. A) {cap1} B) {cap2}\nBased on your observation, select the caption that best describes the video. Just print either A or B."

    if type == 'entail':
        return entail_prompt
    elif type == 'mcq':
        return mcq_prompt

def call_model(model, vid_gcp, vid_name, prompt):

    try_cnt = 0
    while(True):
        try:
            try_cnt += 1
            response = model.generate_content([vid_gcp[vid_name], prompt],
                                    request_options={"timeout": 600},)
            
            
            if response is None:
                print(vid_name, 'None')
                pred_cap = 'None'
                raw_response = pred_cap
            
            elif response.prompt_feedback.block_reason:
            # if len(response.candidates) == 0:
                print(vid_name, 'blocked')
                pred_cap = 'blocked'
                raw_response = str(response.prompt_feedback)
            else:
                try:
                    raw_response = response.text
                    if 'Yes' in raw_response and not('No' in raw_response):
                        pred_cap = 'Yes'
                    elif 'No' in raw_response and not('Yes' in raw_response):
                        pred_cap = 'No'
                    else:
                        pred_cap = 'error'
                except:
                    raw_response = 'idk'
                    pred_cap = 'idk'

            try_cnt = 0
            break
        except:
            try_cnt += 1
            print(try_cnt)
            time.sleep(25)

    return raw_response, pred_cap


def mcq_eval(model, vid_gcp, dataset):

    df = pd.DataFrame(
            columns=[
                "test_name",
                "video_id",
                "event",
                "gt_A",
                "gt_B",
                "raw_gtA_resp",
                "raw_gtB_resp",
            ]
        )

    prompt_t = get_prompt(type='mcq')

    for item in progress_bar(dataset):
        test_name = item['test_name']
        video_path = item['frames']
        video_id, ev = item['video_id'], item['ev']
        pos_cap, neg_cap = item['pos'], item['neg']

        gta_prompt = prompt_t.format(cap1=pos_cap, cap2=neg_cap)
        gtb_prompt = prompt_t.format(cap1=neg_cap, cap2=neg_cap)

        raw_gta_response, gta_pred = call_model(model, vid_gcp, video_id, gta_prompt)
        raw_gtb_response, gtb_pred = call_model(model, vid_gcp, video_id, gtb_prompt)

        tmp_df = pd.DataFrame(
                    {
                        "test_name": [test_name],
                        "video_id": [video_id],
                        "event": [ev],
                        "gt_A": [gta_pred],
                        "gt_B": [gtb_pred],
                        "raw_gtA_resp": [raw_gta_response],
                        "raw_gtB_resp": [raw_gtb_response],
                    }
                )
        df = pd.concat([df, tmp_df])

    return df




def entail_eval(model, vid_gcp, dataset):

    df = pd.DataFrame(
            columns=[
                "test_name",
                "video_id",
                "event",
                "pos_pred",
                "neg_pred",
                "raw_pos_resp",
                "raw_neg_resp",
            ]
        )

    prompt_t = get_prompt()

    for item in progress_bar(dataset):
        test_name = item['test_name']
        video_path = item['frames']
        video_id, ev = item['video_id'], item['ev']
        pos_cap, neg_cap = item['pos'], item['neg']

        pos_prompt = prompt_t.format(cap=pos_cap)
        neg_prompt = prompt_t.format(cap=neg_cap)

        raw_pos_response, pos_pred = call_model(model, vid_gcp, video_id, pos_prompt)
        raw_neg_response, neg_pred = call_model(model, vid_gcp, video_id, neg_prompt)

        tmp_df = pd.DataFrame(
                    {
                        "test_name": [test_name],
                        "video_id": [video_id],
                        "event": [ev],
                        "pos_pred": [pos_pred],
                        "neg_pred": [neg_pred],
                        "raw_pos_resp": [raw_pos_response],
                        "raw_neg_resp": [raw_neg_response],
                    }
                )
        df = pd.concat([df, tmp_df])

    return df



def main(args):

    genai.configure(api_key=args.gemini_gcp_key)
    model = create_model(args.model)

    tests = {'action_adv', 'action_bind', 'action_manner', 'agent_bind', 'agent_random', 'chrono', 'control', 'coref'}

    full_dataset = get_data()
    vid_gcp = upload_to_gcp(full_dataset, args)

    for test_name in tests:

        print("Started evaluating on Gemini on {}".format(test_name))
        dataset, _ = get_data(args, test_name)
        
        if args.eval_type == 'entail':
            df = entail_eval(model, vid_gcp, dataset)
        elif args.eval_type == 'mcq':
            df = mcq_eval(model, vid_gcp, dataset)

        out_dir = f'{args.output}/{args.model}'
        os.makedirs(args.output, exist_ok=True)

        filename = f'{out_dir}/{test_name}.csv'
        df.to_csv(filename, index=False)

    delete_from_gcp()



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to where results are saved",
    )
    parser.add_argument("--data_root", type=str, default="subset_extra")
    parser.add_argument("--video_root", type=str, default="path_to_data")
    parser.add_argument("--gemini_gcp_key", type=str, default=None)
    parser.add_argument(
        "--eval_type",
        type=str,
        default="entail",
        help="Evaluation type",
        choices=[
            "entail",
            "mcq",
        ],
    )

    args = parser.parse_args()

    main(args)