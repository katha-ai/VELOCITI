import torch
from .eval import *
from utils.utils import matrix_dotprod
from fastprogress.fastprogress import progress_bar
import pandas as pd

def negeval(dataloader, model, tokenizer, device, model_type, neg_sampling, args):
    
    """
    the dataloader's dataset has a dependency on which negative to choose, and that is what determines 
    the negative sample for which the accuracy will be calculated.
    """
    df = None
    
    if args.exhaustive_log:
        df = pd.DataFrame(columns=['video_id', 'event', 'pos_score', 'neg_score', 'pos_cap', 'neg_cap'])
    cnt, tot = 0,0
            
    model.eval()
    with torch.no_grad():
        pb = progress_bar(dataloader)
        for item in pb:
            frames = get_video_features(model=model, frames=item['frames'], device=device, model_type=model_type)
            cap = get_cap_features(model=model, cap=item['pos_cap'], tokenizer=tokenizer, device=device, model_type=model_type)
            neg_cap = get_cap_features(model=model, cap=item['neg_cap'], tokenizer=tokenizer, device=device, model_type=model_type)
            
            batch_size = frames.shape[0]
            pos_score = matrix_dotprod(frames, cap)
            neg_score = matrix_dotprod(frames, neg_cap)
            
            cnt += (pos_score > neg_score).sum().item()
            tot += batch_size

            if args.exhaustive_log:
                if neg_sampling == 'coref' or neg_sampling == 'seqence':
                    # item['event'] = list(zip(item['event'][0], item['event'][1], item['event'][2]))
                    item['event'] = item['event'][0]
                    
                    
                tmp_df = pd.DataFrame({
                    "video_id": item['vid_name'],
                    "event": item['event'],
                    "pos_score": pos_score.reshape(-1).cpu().numpy().tolist(),
                    "neg_score": neg_score.reshape(-1).cpu().numpy().tolist(),
                    "pos_cap": item['pos_cap'],
                    "neg_cap": item['neg_cap']
                })
                df = pd.concat([df, tmp_df])
    
    return (cnt/tot), df