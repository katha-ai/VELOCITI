import argparse
import os
import torch
import configs.model_card as models
from model_loaders import *
from dataloaders.control_dataset import controlDataset
from dataloaders.neg_dataset import negDataset
from dataloaders.ivat_dataset import ivatDataset
from eval.video_caption_match import vid_cap_match
from eval.genericeval import genericeval
from eval.negeval import negeval
from utils.utils import create_dataloader, set_seed
import pandas as pd
from tabulate import tabulate

def get_all_metrics(model, tokenizer, transform, model_type, model_name, device):
    metric_dict = {}
    print("\n<----- Caption Matching Task Evaluation ----->\n")
    vc_match_dataset = ivatDataset(data_dict=data_dict, transform=transform)
    vc_match_loader = create_dataloader(dataset=vc_match_dataset, args=args)
    
    vc_match_metric, vc_match_df = vid_cap_match(dataloader=vc_match_loader,
                                    model=model,
                                    tokenizer=tokenizer,
                                    device=device,
                                    model_type=model_type,
                                    args = args)
    
    metric_dict['model'] = model_name
    metric_dict['match_wino_t2v'] = vc_match_metric['video-caption-matching']['wino']['t2v']
    metric_dict['match_wino_v2t'] = vc_match_metric['video-caption-matching']['wino']['v2t']
    metric_dict['match_wino_group'] = vc_match_metric['video-caption-matching']['wino']['group']
    metric_dict['match_ind_t2v'] = vc_match_metric['video-caption-matching']['ind']['t2v']
    metric_dict['match_ind_v2t'] = vc_match_metric['video-caption-matching']['ind']['v2t']
    
    #control task
    print("\n<----- Control Task Evaluation ----->\n")
    control_dataset = controlDataset(data_dict=data_dict,
                                      transform=transform)
    
    control_dataloader = create_dataloader(dataset=control_dataset, args=args)
    
    
    control_metric_v2t, control_v2t_df = genericeval(dataloader=control_dataloader,
                                     model=model,
                                     tokenizer=tokenizer,
                                     task="v2t",
                                     iscontrol=True,
                                     device=device,
                                     model_type=model_type,
                                     args=args)
    
    metric_dict['control_v2t'] = control_metric_v2t
    
    control_metric_t2v, control_t2v_df = genericeval(dataloader=control_dataloader,
                                     model=model,
                                     tokenizer=tokenizer,
                                     task="t2v",
                                     iscontrol=True,
                                     device=device,
                                     model_type=model_type,
                                     args=args)
    
    metric_dict['control_t2v'] = control_metric_t2v
    
    # negatives v2t tasks

    agiden_dataset = negDataset(data_dict=data_dict,
                                     transform=transform,
                                     neg_sampling='ag_iden')
    
    arg0en_dataloader = create_dataloader(dataset=agiden_dataset, args=args)
    
    
    arg0hn_dataset = negDataset(data_dict=data_dict,
                                     transform=transform,
                                     neg_sampling='ag_bind')
    
    arg0hn_dataloader = create_dataloader(dataset=arg0hn_dataset, args=args)
    
    verb_dataset = negDataset(data_dict=data_dict,
                                   transform=transform,
                                   neg_sampling='action_bind')
    
    verb_dataloader = create_dataloader(dataset=verb_dataset, args=args)
    
    manner_dataset = negDataset(data_dict=data_dict,
                                     transform=transform,
                                     neg_sampling='action_mod')
    
    manner_dataloader = create_dataloader(dataset=manner_dataset, args=args)
    
    verb_cot_dataset = negDataset(data_dict=data_dict,
                                        transform=transform,
                                        neg_sampling='action_adv')
    
    verb_cot_dataloader = create_dataloader(dataset=verb_cot_dataset, args=args)
    
    coref_dataset = negDataset(data_dict=data_dict,
                                        transform=transform,
                                        neg_sampling='coref')
    
    coref_dataloader = create_dataloader(dataset=coref_dataset, args=args)
    
    seq_dataset = negDataset(data_dict=data_dict,
                                        transform=transform,
                                        neg_sampling='sequence')
    
    seq_dataloader = create_dataloader(dataset=seq_dataset, args=args)
    
    
    print("\n<----- Agent-Identity-Test Task Evaluation ----->\n")
    ag_iden_metric, ag_iden_df = negeval(dataloader=arg0en_dataloader,
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            model_type=model_type,
                            neg_sampling='ag_iden',
                            args = args)
    
    print("\n<----- Agent-Binding-Test Task Evaluation ----->\n")
    ag_bind_metric, ag_bind_df = negeval(dataloader=arg0hn_dataloader,
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            model_type=model_type,
                            neg_sampling='ag_bind',
                            args = args)
    
    print("\n<----- Action-Binding-Test Task Evaluation ----->\n")
    act_bind_metric, act_bind_df = negeval(dataloader=verb_dataloader,
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            model_type=model_type,
                            neg_sampling='action_bind',
                            args = args)
    
    print("\n<----- Action-Modifier-Test Task Evaluation ----->\n")
    act_mod_metric, act_mod_df = negeval(dataloader=manner_dataloader,
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            model_type=model_type,
                            neg_sampling='action_mod',
                            args = args)

    
    print("\n<----- Action-Adversarial-Test Task Evaluation ----->\n")    
    act_adv_metric, act_adv_df = negeval(dataloader=verb_cot_dataloader,
                                           model=model,
                                           tokenizer=tokenizer,
                                           device=device,
                                           model_type=model_type,
                                           neg_sampling='action_adv',
                                           args=args)

    
    # Coreference Task Evaluation
    print("\n<----- Agent-Coreference Task Evaluation ----->\n")

    coref_metric, coref_df = negeval(dataloader=coref_dataloader,
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            model_type=model_type,
                            neg_sampling='coref',
                            args = args)

    # Sequence Task Evaluation
    
    print("\n<----- Chronology Task Evaluation ----->\n")    
    seq_metric, seq_df = negeval(dataloader=seq_dataloader,
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            model_type=model_type,
                            neg_sampling='sequence',
                            args = args)
    
    
    
    metric_dict['ag_iden'] = ag_iden_metric
    metric_dict['ag_bind'] = ag_bind_metric
    metric_dict['action_bind'] = act_bind_metric
    metric_dict['action_adv'] = act_adv_metric
    metric_dict['action_mod'] = act_mod_metric
    metric_dict['coref'] = coref_metric
    metric_dict['sequence'] = seq_metric
    
    if args.exhaustive_log:
        all_df = {
            'ag_iden': ag_iden_df,
            'ag_bind': ag_bind_df,
            'action_bind': act_bind_df,
            'action_mod': act_mod_df,
            'coref': coref_df,
            'seq': seq_df,
            'control_v2t': control_v2t_df,
            'control_t2v': control_t2v_df,
            'vc_match': vc_match_df,
            'action_adv': act_adv_df
        }
    else:
        all_df = None
    
    return metric_dict, all_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default="clip_B_32", help="Model architecture to be used", choices=["clip_B_32", "clip_L_14", "evaclip_L_14", "siglip_B_16", "siglip_L_16", "negclip_B_32", "clipvip_B_32", "vificlip"])
    parser.add_argument('--cache_dir', default='.hfcache', type=str, help="Directory to where downloaded models are cached")
    parser.add_argument('--output', type=str, default="output/", help="Directory to where results are saved")
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--all', action="store_true", default=False, help="Whether to test all the pretrained models in the paper")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--exhaustive_log', action="store_true", default=True, help="wether to log the results of each sample")

    
    
    args = parser.parse_args()
    
    device = torch.device(args.device)

    data_dict = {
        "vidsitu_dict_path"  : f'{args.data_root}/vidsitu_dict.json',
        "frames_path"        : f'{args.data_root}/frames',
        "agent_iden_caps"    : f'{args.data_root}/agent_iden.json',
        "agent_bind_caps"    : f'{args.data_root}/agent_bind.json',
        "action_bind_caps"   : f'{args.data_root}/action_bind.json',
        "action_mod_caps"    : f'{args.data_root}/action_mod.json',
        "control_neg_caps"   : f'{args.data_root}/control.json',
        "coref_caps"         : f'{args.data_root}/coref.json',
        "seq_caps"           : f'{args.data_root}/sequence.json',
        "action_adv_caps"    : f'{args.data_root}/action_adv.json'}
    
    os.makedirs(args.output, exist_ok=True)
    set_seed(args.seed)
    
    if args.all:
        all_models_metrics = []
        
        for model in models.keys():
            args.model = model
            print(f"Evaluating {model}")
            if model in ["clip_B_32", "clip_L_14", "evaclip_L_14", "siglip_B_16", "siglip_L_16"]:
                model, tokenizer, transform = load_open_clip_model(args, device=device)
                model_type = 'open_clip'
                
            elif model == "negclip_B_32":
                model, tokenizer, transform = load_neg_clip_model(args, device=device)
                model_type = 'neg_clip'
            
            elif model == "clipvip_B_32":
                model, tokenizer, transform = load_clip_vip_model(args, device=device)
                model_type = 'clip_vip'
            
            elif model == "vificlip":
                model, tokenizer, transform = load_vifi_clip_model(args, device=device)
                model_type = 'vificlip'
            
            metric_dict, all_df = get_all_metrics(model, tokenizer, transform, model_type=model_type, model_name=args.model, device=device)
            
            all_models_metrics.append(metric_dict)
            
            if args.exhaustive_log:
                csv_path = join(args.output, args.model)
            
                if not os.path.exists(csv_path):
                    os.makedirs(csv_path)
                
                for key in all_df.keys():
                    if all_df[key] is not None:
                        all_df[key].to_csv(f"{csv_path}/{key}.csv", index=False)
            del model
            del tokenizer
            del transform
        
        df = pd.DataFrame(all_models_metrics)
        print(f"Results of all models are as follows:")
        print(tabulate(df, headers='keys', tablefmt='psql'))
        print(f"Results saved to {args.output}/all_models.csv")
        df.to_csv(f"{args.output}/all_models.csv", index=False)
    
    else:
        print(f"Evaluating {args.model}")
        model = args.model
        
        if model in ["clip_B_32", "clip_L_14", "evaclip_L_14", "siglip_B_16", "siglip_L_16"]:
                model, tokenizer, transform = load_open_clip_model(args, device=device) 
                model_type = 'open_clip'
        elif model == "negclip_B_32":
            model, tokenizer, transform = load_neg_clip_model(args, device=device)
            model_type = 'neg_clip'
        elif model == "clipvip_B_32":
            model, tokenizer, transform = load_clip_vip_model(args, device=device)
            model_type = 'clip_vip'
        
        elif model == "vificlip":
            model, tokenizer, transform = load_vifi_clip_model(args, device=device)
            model_type = 'vificlip'
            
        metric_dict, all_df = get_all_metrics(model, tokenizer, transform, model_type=model_type, model_name=args.model, device=device)
        
    
        if args.exhaustive_log:
            csv_path = join(args.output, args.model)
            
            if not os.path.exists(csv_path):
                os.makedirs(csv_path)
            
            for key in all_df.keys():
                if all_df[key] is not None:
                    all_df[key].to_csv(f"{csv_path}/{key}.csv", index=False)
        
        df = pd.DataFrame([metric_dict])
        print(f"Results of the {args.model} model are as follows:")
        print(tabulate(df, headers='keys', tablefmt='psql'))
        print(f"Results saved to {args.output}/{model_type}.csv")
        df.to_csv(f"{args.output}/{args.model}.csv", index=False)