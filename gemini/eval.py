import time
import argparse

import google.generativeai as genai
import pandas as pd

import os
from os.path import join
from tqdm import tqdm


def upload_to_gcp(df_data, args):
    vid_gcp = {}
    print("Started upload to GCP")

    for index, row in tqdm(df_data.iterrows(), total=len(df_data)):
        vid_name = row['video_id']
        vid_path = join(args.video_root, vid_name + '.mp4')
        print(vid_path)
        vid_gcp[vid_name] = genai.upload_file(path=vid_path)

    print("Completed upload to GCP")
    return vid_gcp

def delete_from_gcp(vid_gcp):
    print("Deleting videos from GCP")
    for k, v in tqdm(vid_gcp.items(), total=len(vid_gcp)):
        genai.delete_file(v.name)
    print("Deleting done")


def create_model():
    generation_config = {
    "temperature": 0,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
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

    model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
                                generation_config=generation_config,
                                safety_settings=safety_settings)

    return model



def call_model(model, vid_gcp, prompt, row, video_root):
    cap1, cap2 = row['cap1'], row['cap2']
    gt = row['gt']
    vid_name = row['video_id']

    prompt = prompt.format(cap1, cap2)
 
    vid_path = join(video_root, vid_name + '.mp4')

    raw_response, prediction = None, None

    sleep_time = 2
    while(True):
        try:
            # generation_config = genai.GenerationConfig(top_k=1, temperature=0)
            response = model.generate_content([vid_gcp[vid_name], prompt],
                                    request_options={"timeout": 600})
            
            if response.prompt_feedback.block_reason == 2:
                print(vid_name, 'blocked')
                pred_cap = 'blocked'
                raw_response = pred_cap
            elif response is None:
                print(vid_name, 'None')
                pred_cap = 'None'
                raw_response = pred_cap
            else:
                pred_cap = response.text.split(': (')[0]
                if 'A' in pred_cap:
                    pred_cap = 'cap1'
                elif 'B' in pred_cap:
                    pred_cap = 'cap2'
                else:
                    print(vid_name, 'error')
                    pred_cap = response.text
                raw_response = response.text

            prediction = pred_cap
            break
        except:
            time.sleep(sleep_time)
            sleep_time = sleep_time*2 if sleep_time < 32 else sleep_time+30

    return raw_response, prediction


def calc_accuracy(df_csv):
    cnt, tot_cnt = 0,0
    correct = []
    for index, row in df_csv.iterrows():
        if row['pred'] == 'cap1' or row['pred'] == 'cap2': 
            tot_cnt += 1
            if row['pred'] == row['gt']:
                cnt += 1
                correct.append(1)
            else:
                correct.append(0)
        else:
            correct.append('NA')
    accuracy = round((cnt/tot_cnt) * 100, 2)

    print("correct_count: ", cnt, end=' ')
    print("total_count: ", tot_cnt, end=' ')
    print("Accuracy: ", round((cnt/tot_cnt) * 100, 2))

    return cnt, tot_cnt, accuracy, correct



def main(args, data_dict):

    input_file_path = data_dict[args.test]
    df_data = pd.read_csv(input_file_path)
    print("Length of the csv:", len(df_data))

    args.gemini_gcp_key = 'AIzaSyBGU2lpmIwprPCw7wYkB82yYpwoLUQ4p10'
    genai.configure(api_key=args.gemini_gcp_key)

    model = create_model()

    vid_gcp = upload_to_gcp(df_data, args)

    print("Started evaluating on Gemini")
    pred_response = {'video_id': [], 'pred': [], 'gt': [], 'response':[]}

    prompt = "Carefully watch the video and pay attention to the sequence of events, the details and actions of persons. Based on your observation, select the caption that best describes the video. Just print either A) or B). A) {} B) {} ASSISTANT: Best Caption : ("

    for index, row in tqdm(df_data.iterrows(), total=len(df_data)):
        cap1, cap2 = row['cap1'], row['cap2']
        gt = row['gt']
        vid_name = row['video_id']

        if vid_name in pred_response:
            continue

        raw_response, prediction = call_model(model, vid_gcp, prompt, row, args.video_root)

        pred_response['video_id'].append(vid_name)
        pred_response['pred'].append(prediction)
        pred_response['gt'].append(row['gt'])
        pred_response['response'].append(raw_response)

        # Intermediate results store
        df_csv = pd.DataFrame(pred_response)
        df_csv.to_csv(join(args.output_dir, args.test), index=False)

    cnt, tot_cnt, accuracy, pred_response['correct'] = calc_accuracy(df_csv)
    
    pred_response['video_id'].append('')
    pred_response['pred'].append('')
    pred_response['gt'].append('')
    pred_response['correct'].append('')
    pred_response['response'].append('')

    pred_response['video_id'].append("correct: {}".format(cnt))
    pred_response['pred'].append("total: {}".format(tot_cnt))
    pred_response['gt'].append('Acc: {}%'.format(accuracy))
    pred_response['correct'].append('')
    pred_response['response'].append('')


    print("Storing results at {}".format(join(args.output_dir, args.test)))
    df_csv = pd.DataFrame(pred_response)
    df_csv.to_csv(join(args.output_dir, args.test), index=False)

    delete_from_gcp(vid_gcp)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test",
        type=str,
        default="ag_iden",
        help="test name",
        choices=[
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
        "--output_dir",
        type=str,
        default="output/",
        help="Directory to where results are saved",
    )
    parser.add_argument("--data_root", type=str, default="./gemini_data")
    parser.add_argument("--video_root", type=str, default="./data/videos/velociti_videos_10s")
    parser.add_argument("--gemini_gcp_key", type=str)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data_dict = {
        "ag_iden": f"{args.data_root}/agent_iden.csv",
        "ag_bind": f"{args.data_root}/agent_bind.csv",
        "action_bind": f"{args.data_root}/action_bind.csv",
        "action_mod": f"{args.data_root}/action_mod.csv",
        "coref": f"{args.data_root}/coref.csv",
        "sequence": f"{args.data_root}/sequence.csv",
        "action_adv": f"{args.data_root}/action_adv.csv",
    }

    main(args, data_dict)