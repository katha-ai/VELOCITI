import torch
from .eval import *
from utils.utils import matrix_dotprod
from fastprogress.fastprogress import master_bar, progress_bar
import pandas as pd

def vid_cap_match(dataloader, model, tokenizer, device, model_type, args):
    
    metric = {'video-caption-matching': {}}
    gt_cnt, gv_cnt, group_cnt, tot = 0,0,0,0
    ind_gt_cnt, ind_gv_cnt = 0,0
    
    exhasutive_log = args.exhaustive_log
    
    df = None
    if exhasutive_log:
        df = pd.DataFrame(columns=['video_id', 'event', 'v1_c1_score', 'v1_c5_score', 'v5_c1_score', 'v5_c5_score', 'pos_cap_ev1', 'pos_cap_ev5'])
    
    model.eval()
    with torch.no_grad():
        pb = progress_bar(dataloader)
        for item in pb:
            pb.comment = f"Performing Video Caption Matching Evaluation"
            frames_ev1 = item['frames_ev1']
            frames_ev5 = item['frames_ev5']
            pos_cap_ev1 = item['pos_cap_ev1']
            pos_cap_ev5 = item['pos_cap_ev5']
            
            batch_size = frames_ev1.shape[0]
            
            ev1_frame_feats = get_video_features(model=model, frames=frames_ev1, device=device, model_type=model_type)
            ev5_frame_feats = get_video_features(model=model, frames=frames_ev5, device=device, model_type=model_type)
            
            ev1_cap_feats = get_cap_features(model=model, cap=pos_cap_ev1, tokenizer=tokenizer, device=device, model_type=model_type)
            ev5_cap_feats = get_cap_features(model=model, cap=pos_cap_ev5, tokenizer=tokenizer, device=device, model_type=model_type)
            
            v1_c1_score = matrix_dotprod(ev1_frame_feats, ev1_cap_feats)
            v1_c5_score = matrix_dotprod(ev1_frame_feats, ev5_cap_feats)    
            v5_c1_score = matrix_dotprod(ev5_frame_feats, ev1_cap_feats)
            v5_c5_score = matrix_dotprod(ev5_frame_feats, ev5_cap_feats)
            
            gt_res = torch.logical_and((v1_c1_score > v5_c1_score),(v5_c5_score > v1_c5_score))
            gv_res = torch.logical_and((v1_c1_score > v1_c5_score),(v5_c5_score > v5_c1_score))
            group_res = torch.logical_and(gt_res,gv_res)

            gt_cnt += gt_res.sum()
            gv_cnt += gv_res.sum()
            group_cnt += group_res.sum()

            ind_gt_cnt += (v1_c1_score > v5_c1_score).sum() + (v5_c5_score > v1_c5_score).sum()
            ind_gv_cnt += (v1_c1_score > v1_c5_score).sum() + (v5_c5_score > v5_c1_score).sum()

            tot += batch_size
            
            if exhasutive_log:
                temp_df = pd.DataFrame({
                    "video_id": item['vid_name'],
                    "event": item['event'],
                    "v1_c1_score": v1_c1_score.reshape(-1).detach().cpu().tolist(),
                    "v1_c5_score": v1_c5_score.reshape(-1).detach().cpu().tolist(),
                    "v5_c1_score": v5_c1_score.reshape(-1).detach().cpu().tolist(),
                    "v5_c5_score": v5_c5_score.reshape(-1).detach().cpu().tolist(),
                    "pos_cap_ev1": item['pos_cap_ev1'],
                    "pos_cap_ev5": item['pos_cap_ev5']
                })
                
                df = pd.concat([df, temp_df], ignore_index=True)
                
    

    for task in metric:
        metric[task]['wino'] = {
            't2v': (gt_cnt / tot).item(),
            'v2t': (gv_cnt / tot).item(),
            'group': (group_cnt / tot).item()
        }

        metric[task]['ind'] = {
            't2v': (ind_gt_cnt / (2 * tot)).item(),
            'v2t': (ind_gv_cnt / (2 * tot)).item()
        }


    return metric, df