import torch
from .eval import *
from utils.utils import matrix_dotprod
from fastprogress import progress_bar
import pandas as pd

def genericeval(dataloader, model, tokenizer, task, iscontrol, device, model_type, args):
    # iscontrol
    # task -> t2v or v2t
    
    # iscontrol=False + task=v2t is the setup for "Hard Negatives" task.
    
    cnt, tot = 0,0
    
    df = None
    
    if args.exhaustive_log:
        df = pd.DataFrame(columns=['video_id', 'event', 'pos_score', 'neg_score', 'pos_cap', 'neg_cap'])
    model.eval()
    with torch.no_grad():
        pb = progress_bar(dataloader)
        for item in pb:
            if iscontrol and task=='t2v':
                pb.comment = f"Performing Control Task Evaluation for Text-to-Video Task" 
                frames = get_video_features(model=model, frames=item['frames'], device=device, model_type=model_type)
                neg_frames = get_video_features(model=model, frames=item['neg_frames'], device=device, model_type=model_type)
                
                cap = get_cap_features(model=model, cap=item['pos_cap'], tokenizer=tokenizer, device=device, model_type=model_type)
                batch_size = frames.shape[0]
                
                pos_score = matrix_dotprod(frames, cap)
                neg_score = matrix_dotprod(neg_frames, cap)
                
                cnt += (pos_score > neg_score).sum().item()
                tot += batch_size
                
            
            elif iscontrol and task=="v2t":
                pb.comment = f"Performing Control Task Evaluation for Video-to-Text Task" 
                frames = get_video_features(model=model, frames=item['frames'], device=device, model_type=model_type)
                cap = get_cap_features(model=model, cap=item['pos_cap'], tokenizer=tokenizer, device=device, model_type=model_type)
                neg_cap = get_cap_features(model=model, cap=item['neg_cap'], tokenizer=tokenizer, device=device, model_type=model_type)
                
                if frames is None:
                    print("what now?")
                batch_size = frames.shape[0]
                
                pos_score = matrix_dotprod(frames, cap)
                neg_score = matrix_dotprod(frames, neg_cap)
                
                cnt += (pos_score > neg_score).sum().item()
                tot += batch_size
                
            else:
                raise ValueError("Invalid task and control flag combination specified.")

            if args.exhaustive_log:
                    tmp_df = pd.DataFrame({
                        "video_id": item['vid_name'],
                        "event": item['event'],
                        "pos_score": pos_score.reshape(-1).detach().cpu().tolist(),
                        "neg_score": neg_score.reshape(-1).detach().cpu().tolist(),
                        "pos_cap": item['pos_cap'],
                        "neg_cap": item['neg_cap']
                    })
                    df = pd.concat([df, tmp_df])
    
    return (cnt/tot), df