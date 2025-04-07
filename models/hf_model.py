import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


def init_model(pretrained="llava-hf/llava-onevision-qwen2-7b-ov-hf", cache_dir='.'):

    model = AutoModelForImageTextToText.from_pretrained(pretrained,
                                                                   torch_dtype=torch.float16,
                                                                   low_cpu_mem_usage=True,
                                                                   attn_implementation="flash_attention_2",
                                                                   device_map='auto',
                                                                   cache_dir=cache_dir,
                                                                   load_in_4bit=False)
    processor = AutoProcessor.from_pretrained(pretrained)
    return model, processor
