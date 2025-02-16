import os
import torch 
import numpy as np
from jax import grad, vmap
from tqdm import tqdm
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from huggingface_hub import login
from data.serialize import serialize_arr, deserialize_str, SerializerSettings

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("Hugging Face authentication token is missing. Please set the HF_TOKEN environment variable.")

login(HF_TOKEN) 

DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

loaded = {}

def deepseek_model_string(model_name):
    """Handles both standard DeepSeek models and DeepSeek-R1."""
    if model_name.lower() == "deepseek-r1":
        return "deepseek-ai/DeepSeek-R1"
    return f"deepseek-ai/deepseek-{model_name}"

def get_tokenizer(model_name):
    """Fetches tokenizer for DeepSeek models, including R1."""
    tokenizer = AutoTokenizer.from_pretrained(
        deepseek_model_string(model_name),
        use_fast=False,
        token=HF_TOKEN  
    )

    special_tokens_dict = {}
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def get_model_and_tokenizer(model_name, cache_model=False):
    """Loads the DeepSeek model and tokenizer, supporting DeepSeek-R1."""
    if model_name in loaded:
        return loaded[model_name]

    tokenizer = get_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        deepseek_model_string(model_name),
        device_map="auto",   
        torch_dtype=torch.float16,
        trust_remote_code=True,  # Needed for DeepSeek-R1
        token=HF_TOKEN  
    )
    model.eval()

    if cache_model:
        loaded[model_name] = model, tokenizer

    return model, tokenizer

def tokenize_fn(text, model_name):
    """Tokenizes input text using the appropriate tokenizer."""
    tokenizer = get_tokenizer(model_name)
    return tokenizer(text, return_tensors="pt")

def deepseek_nll_fn(model_name, input_arr, target_arr, settings: SerializerSettings, transform, count_seps=True, temp=1, cache_model=True):
    model, tokenizer = get_model_and_tokenizer(model_name, cache_model=cache_model)

    input_str = serialize_arr(vmap(transform)(input_arr), settings)
    target_str = serialize_arr(vmap(transform)(target_arr), settings)
    full_series = input_str + target_str
    
    batch = tokenizer(
        [full_series], 
        return_tensors="pt",
        add_special_tokens=True
    )
    batch = {k: v.cuda() for k, v in batch.items()}

    with torch.no_grad():
        out = model(**batch)

    good_tokens_str = list("0123456789" + settings.time_sep)
    good_tokens = [tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]
    bad_tokens = [i for i in range(len(tokenizer)) if i not in good_tokens]
    out['logits'][:, :, bad_tokens] = -100

    input_ids = batch['input_ids'][0][1:]
    logprobs = torch.nn.functional.log_softmax(out['logits'], dim=-1)[0][:-1]
    logprobs = logprobs[torch.arange(len(input_ids)), input_ids].cpu().numpy()
    
    input_len = len(tokenizer([input_str], return_tensors="pt")['input_ids'][0]) - 2
    logprobs = logprobs[input_len:]
    BPD = -logprobs.sum() / len(target_arr)

    transformed_nll = BPD - settings.prec * np.log(settings.base)
    avg_logdet_dydx = np.log(vmap(grad(transform))(target_arr)).mean()
    
    return transformed_nll - avg_logdet_dydx

def deepseek_completion_fn(
    model_name,
    input_str,
    steps,
    settings,
    batch_size=5,
    num_samples=20,
    temp=0.9, 
    top_p=0.9,
    cache_model=True
):
    avg_tokens_per_step = len(tokenize_fn(input_str, model_name)['input_ids'][0]) / len(input_str.split(settings.time_sep))
    max_tokens = int(avg_tokens_per_step * steps)
    
    model, tokenizer = get_model_and_tokenizer(model_name, cache_model=cache_model)

    gen_strs = []
    for _ in tqdm(range(num_samples // batch_size)):
        batch = tokenizer(
            [input_str], 
            return_tensors="pt",
        )
        batch = {k: v.repeat(batch_size, 1) for k, v in batch.items()}
        batch = {k: v.cuda() for k, v in batch.items()}
        num_input_ids = batch['input_ids'].shape[1]

        good_tokens_str = list("0123456789" + settings.time_sep)
        good_tokens = [tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]
        bad_tokens = [i for i in range(len(tokenizer)) if i not in good_tokens]

        generate_ids = model.generate(
            **batch,
            do_sample=True,
            max_new_tokens=max_tokens,
            temperature=temp, 
            top_p=top_p, 
            bad_words_ids=[[t] for t in bad_tokens],
            renormalize_logits=True,
        )
        gen_strs += tokenizer.batch_decode(
            generate_ids[:, num_input_ids:],
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
    
    return gen_strs
