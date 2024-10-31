# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List
from tqdm import tqdm

from transformers import AutoTokenizer, LlamaTokenizer
from src.utils.str import is_empty

PROMPT_DICT = {
    "alpaca":{
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        )
    }
}


def check_and_exit(*args, **kwargs):
    print(*args, **kwargs)
    exit(12)


def multiprocess_call(func, input_list, kwarg_list=None, num_threads=8):
    from multiprocessing import Pool

    pool = Pool(num_threads)
    results = []
    if not kwarg_list:
        kwarg_list = [{}] * len(input_list)
    for inp, kwargs in zip(input_list, kwarg_list):
        results.append(pool.apply_async(func, inp, kwargs))
    pool.close()
    pool.join()

    return [res.get() for res in results]


def build_qwen_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("/home/chengan.lk/liuk/projects/tq-llm/weights/Qwen2-7B-Chat", trust_remote_code=True)
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"

    return tokenizer


def build_llama_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("/home/chengan.lk/liuk/projects/tq-llm/weights/Llama-2-7b-chat-hf", trust_remote_code=True)

    return tokenizer
