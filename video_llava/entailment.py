import torch

def get_entail(logits, input_ids, tokenizer):
    
    logits = torch.nn.functional.softmax(logits, dim=-1)
    token_id_yes = tokenizer.encode('Yes', add_special_tokens = False)[0]
    token_id_no  = tokenizer.encode('No', add_special_tokens = False)[0]
    print(logits[0][-1][token_id_yes],logits[0][-1][token_id_no])
    score = logits[0][-1][token_id_yes] / (logits[0][-1][token_id_yes] + logits[0][-1][token_id_no])
    return torch.tensor([score])


def entail_batch(model, processor, captions, clips):
    template = """USER: <video>Carefully watch the video and pay attention to the sequence of events, the details and actions of persons. Based on your observation, does the given video entail the caption?
Caption: {}
ASSISTANT:"""

    # token_id_yes = processor('Yes').input_ids[0,1]
    # token_id_no = processor('No').input_ids[0,1]

    prompts = list(map(template.format,captions))
    print(prompts)
    inputs = processor(text=prompts, videos=clips, padding=True, return_tensors="pt").to('cuda')

    with torch.no_grad():
        out = model(**inputs)

    print(inputs.keys())
    s = get_entail(out.logits, inputs.input_ids, processor.tokenizer)
    return s