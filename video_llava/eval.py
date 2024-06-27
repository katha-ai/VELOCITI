import sys
import os
import csv
import numpy as np
import argparse
from tqdm import tqdm
import gc
import torch

from video_llava.entailment import *
from video_llava.data_utils import *

from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration, VideoLlavaConfig


def wino(va_ca_score,va_cb_score,vb_ca_score,vb_cb_score):
    gt_res = torch.logical_and((va_ca_score > vb_ca_score),(vb_cb_score > va_cb_score))
    gv_res = torch.logical_and((va_ca_score > va_cb_score),(vb_cb_score > vb_ca_score))
    group_res = torch.logical_and(gt_res,gv_res)

    gt_cnt = gt_res.sum()
    gv_cnt = gv_res.sum()
    group_cnt = group_res.sum()

    ind_gt_cnt = (va_ca_score > vb_ca_score).sum() + (vb_cb_score > va_cb_score).sum()
    ind_gv_cnt = (va_ca_score > va_cb_score).sum() + (vb_cb_score > vb_ca_score).sum()
    return torch.tensor([gt_cnt,gv_cnt,group_cnt,ind_gt_cnt,ind_gv_cnt])


def ivat(args, data_dict):
    dataset, dataloader = get_data_ivat(args, data_dict)

    model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf",cache_dir=args.cache_dir,device_map='auto')
    processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf",cache_dir=args.cache_dir)

    inp_filename = '{}/{}_scores.csv'.format(args.output,args.test)
    fields = ['video_id','va_ca','va_cb','vb_ca','vb_cb','cap_a','cap_b']

    res = torch.zeros(5)
    tot = 0

    with open(inp_filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fields)

        for i, (data_batch) in enumerate(tqdm(dataloader)):
            vid_name, event = data_batch['vid_name'], data_batch['event']
            frames_a_paths, frames_b_paths = data_batch['frames_ev1'], data_batch['frames_ev5']
            cap_a, cap_b = data_batch['pos_cap_ev1'], data_batch['pos_cap_ev5']

            clips_a = list(map(read_video_pyav,frames_a_paths))
            clips_b = list(map(read_video_pyav,frames_b_paths))

            va_ca = entail_batch(model, processor, cap_a, clips_a).cpu()
            va_cb = entail_batch(model, processor, cap_b, clips_a).cpu()
            vb_ca = entail_batch(model, processor, cap_a, clips_b).cpu()
            vb_cb = entail_batch(model, processor, cap_b, clips_b).cpu()

            res += wino(va_ca,va_cb,vb_ca,vb_cb)
            tot += va_ca.shape[0]
            
            writer.writerows(list(zip(vid_name,va_ca.tolist(),va_cb.tolist(),vb_ca.tolist(),vb_cb.tolist(),cap_a,cap_b)))

            gc.collect()
            torch.cuda.empty_cache()

        w = torch.tensor([tot,tot,tot,tot*2,tot*2])
        results = res/w
        writer.writerow(('t2v','v2t','group','ind_t2v','ind_v2t','total',None))
        writer.writerow((*results.tolist(),w[0],None))




def text2video(args, data_dict):

    dataset, dataloader = get_data_control(args, data_dict)

    model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf",cache_dir=args.cache_dir,device_map='auto')
    processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf",cache_dir=args.cache_dir)

    cor,tot = 0,0

    inp_filename = '{}/{}_scores.csv'.format(args.output,args.test)
    fields = ['video_id','ev','pos_score','neg_score','neg_video_id']
 
    with open(inp_filename, 'w') as csvfile:
        print(inp_filename)
        writer = csv.writer(csvfile)
        writer.writerow(fields)

        for i, (data_batch) in enumerate(tqdm(dataloader)):
            vid_name, ev = data_batch['vid_name'], data_batch['event']
            frames, neg_frames = data_batch['frames'], data_batch['neg_frames']
            cap = data_batch['pos_cap']

            pos_clips = list(map(read_video_pyav,frames))
            neg_clips = list(map(read_video_pyav,neg_frames))

            pos_out = entail_batch(model, processor, cap, pos_clips)
            neg_out = entail_batch(model, processor, cap, neg_clips)

            cor += (pos_out > neg_out).sum()
            writer.writerows(list(zip(vid_name,ev,pos_out.tolist(),neg_out.tolist(),neg_frames)))

            tot += pos_out.shape[0]

            gc.collect()
            torch.cuda.empty_cache()

            print('Accuracy at {}:'.format(i), cor,'/',tot)

        acc = cor/tot
        writer.writerow(('Total','Accuracy',None,None,None))
        writer.writerow((tot,acc,None,None,None))



def video2text(args, data_dict):
    print(args.test)

    if args.test == 'control_v2t':
        dataset, dataloader = get_data_control(args, data_dict)
    else:
        dataset, dataloader = get_data_neg(args, data_dict, args.test)

    model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf",cache_dir=args.cache_dir,device_map='auto')
    processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf",cache_dir=args.cache_dir)

    cor,tot = 0,0

    inp_filename = '{}/{}_scores.csv'.format(args.output,args.test)
    fields = ['video_id','ev','pos_score','neg_score','pos_cap','neg_cap']
 
    with open(inp_filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fields)

        for i, (data_batch) in enumerate(tqdm(dataloader)):

            vid_name, ev = data_batch['vid_name'], data_batch['event']
            frames = data_batch['frames']
            pos_cap, neg_cap = data_batch['pos_cap'], data_batch['neg_cap']

            clips = list(map(read_video_pyav,frames))
            
            pos_out = entail_batch(model, processor, pos_cap, clips)
            neg_out = entail_batch(model, processor, neg_cap, clips)

            cor += (pos_out > neg_out).sum()
            writer.writerows(list(zip(vid_name,ev,pos_out.tolist(),neg_out.tolist(),pos_cap,neg_cap)))

            tot += pos_out.shape[0]

            gc.collect()
            torch.cuda.empty_cache()

            print('Accuracy at {}:'.format(i), cor,'/',tot)
            
        acc = cor/tot
        writer.writerow(('Total','Accuracy',None,None,None,None))
        writer.writerow((tot,acc,None,None,None,None))