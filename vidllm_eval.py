import os
from fastprogress.fastprogress import progress_bar
import pandas as pd
import argparse
import torch
import torch.nn.functional as F

from utils.clip_utils import process_video
from models.hf_model import init_model
from data import get_data


def get_prompt(type='entailment'):

    entail_prompt = "Carefully watch the video and pay attention to the sequence of events, the details and actions of persons.\nHere is a caption that describes the video: {caption}\nBased on your observation, does the given video entail the caption? Just answer with either Yes or No. "

    mcq_prompt = "Carefully watch the video and pay attention to the sequence of events, the details and actions of persons. Here are two captions that describe the video. A) {cap1} B) {cap2}\nBased on your observation, select the caption that best describes the video. Just print either A or B."
    
    if type == 'entailment':
        return entail_prompt
    elif type == 'mcq':
        return mcq_prompt
    

def entail_score(logits, processor):

    logits = F.softmax(logits, dim=-1)
    token_id_yes = processor.tokenizer.encode('Yes', add_special_tokens = False)[0]
    token_id_no  = processor.tokenizer.encode('No', add_special_tokens = False)[0]

    scores = logits[:,token_id_yes] / (logits[:,token_id_yes] + logits[:,token_id_no])
    return scores


@torch.no_grad
def inference(model, processor, clips, captions, device, max_length=1, type='entail'):

    conversations = [
        [{
        "role": "user",
        "content": [
            {"type": "video"},
            {"type": "text", "text": cap}
            ],},] for cap in captions]
        
    prompts = [processor.apply_chat_template(conv, add_generation_prompt=True) for conv in conversations]

    # clips = [process_video(vid_path) for vid_path in video_paths]
    
    inputs_video = processor(text=prompts, videos=clips, padding=True, return_tensors="pt").to(model.device, torch.float16)
    output_ids = model.generate(**inputs_video,
                            max_new_tokens=max_length,
                            do_sample=False,
                            output_scores=True,
                            return_dict_in_generate=True)
    
    outputs = processor.decode(output_ids['sequences'][0][-1], skip_special_tokens=True)

    if type == 'entail':
        entailment_scores = entail_score(output_ids['scores'][0], processor)
        return entailment_scores
    elif type == 'mcq':
        return outputs
        

def entail_eval(model, processor, dataloader, device):

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
        video_path = item['frames']
        video_id, ev = item['video_id'], item['ev']
        pos_cap, neg_cap = item['pos'], item['neg']

        clips = [process_video(vid_path) for vid_path in video_path]

        prompt = get_prompt()
        pos_caps = [prompt.format(caption=cap) for cap in pos_cap]
        neg_caps = [prompt.format(caption=cap) for cap in neg_cap]

        pos_score = inference(model, processor, clips, pos_caps, device)
        neg_score = inference(model, processor, clips, neg_caps, device)

        cnt += (pos_score > neg_score).sum().item()
        tot += len(pos_caps)

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


def mcq_eval(model, processor, dataloader, device):

    df = pd.DataFrame(
            columns=[
                "test_name",
                "video_id",
                "event",
                "gt_A",
                "gt_B",
            ]
        )

    for item in progress_bar(dataloader):
        test_name = item['test_name']
        video_path = item['frames']
        video_id, ev = item['video_id'], item['ev']
        pos_cap, neg_cap = item['pos'], item['neg']

        clips = [process_video(vid_path) for vid_path in video_path]

        prompt = get_prompt(type='mcq')
        gt_a, gt_b = [], []

        for i in range(len(pos_cap)):
            gt_a.append(prompt.format(cap1=pos_cap[i], cap2=neg_cap[i]))
            gt_b.append(prompt.format(cap1=neg_cap[i], cap2=pos_cap[i]))

        gt_a_preds = inference(model, processor, clips, gt_a, device, type='mcq')
        gt_b_preds = inference(model, processor, clips, gt_b, device, type='mcq')

        tmp_df = pd.DataFrame(
                    {
                        "test_name": item["test_name"],
                        "video_id": video_id,
                        "event": ev,
                        "gt_A": gt_a_preds,
                        "gt_B": gt_b_preds,
                    }
                )
        df = pd.concat([df, tmp_df])
        df.to_csv('temp.csv', index=False)

    return None, df



def main(args):

    device = torch.device(args.device)

    model, processor = init_model(pretrained=args.model, cache_dir=args.cache_dir)
    
    tests = {'action_adv', 'action_bind', 'action_manner', 'agent_bind', 'agent_random', 'chrono', 'control', 'coref'}

    for test_name in tests:
        print('Evaluating',test_name)
        dataset, dataloader = get_data(args, test_name)
        if args.eval_type == 'entail':
            acc, df = entail_eval(model, processor, dataloader, device)
        elif args.eval_type == 'mcq':
            acc, df = mcq_eval(model, processor, dataloader, device)

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
        default="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        help="HF Model to be used",
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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--exhaustive_log",
        action="store_true",
        default=True,
        help="wether to log the results of each sample",
    )
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