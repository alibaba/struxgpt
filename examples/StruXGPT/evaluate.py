# Copyright (c) Alibaba, Inc. and its affiliates.
import random
import argparse
import os
import os.path as osp
import json
from tqdm import tqdm
from glob import glob
import time
import re
from typing import List, Dict, Union
import numpy as np
import string

import sys
sys.path.append('./')

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from vllm import LLM, SamplingParams

from src.utils.io import load_auto, write_auto
from src.utils.util import check_and_exit
from src.models.struxgpt_base import PreTrainedLLM


def get_longbench_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path):
    from third_party.LongBench.pred import build_chat, post_process
    model = LLM(model2path[model_name], trust_remote_code=True, gpu_memory_utilization=0.6)
    tokenizer = model.get_tokenizer()
    data_to_infer = []
    for json_obj in data:
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        data_to_infer.append(prompt)

    bs = 8
    sampling_kwargs = SamplingParams(temperature=0.0, stop=['</s>', '<|im_end|>'], max_tokens=max_gen)
    pred_list = []
    pbar = range(0, len(data_to_infer), bs)
    pbar = tqdm(pbar) if rank in [0, -1] else pbar
    for i in pbar:
        outputs = model.generate(data_to_infer[i:i+bs], sampling_kwargs, use_tqdm=False)
        pred_list.extend([output.outputs[0].text for output in outputs])

    with open(out_path, "a", encoding="utf-8") as f:
        for pred, json_obj in zip(pred_list, data):
            pred = post_process(pred, model_name)
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')

    if world_size > 1:
        dist.destroy_process_group()


def eval_longbench():
    ## Prediction, borrowed from https://github.com/THUDM/LongBench/blob/main/pred.py
    sys.path.append('./third_party/LongBench')
    from third_party.LongBench.pred import seed_everything
    from third_party.LongBench.eval import scorer

    data_root = 'third_party/LongBench'

    seed_everything(42)
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    model2path = load_auto(f"{data_root}/config/model2path.json")
    model2maxlen = load_auto(f"{data_root}/config/model2maxlen.json")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]
    datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "passage_count", "passage_retrieval_en",]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = load_auto(f"{data_root}/config/dataset2prompt.json")
    dataset2maxlen = load_auto(f"{data_root}/config/dataset2maxlen.json")
    # predict on each dataset
    os.makedirs(f"{data_root}/pred_s", exist_ok=True)
    save_dir = f"{data_root}/pred_s/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    for dataset in datasets[:0]:
        data = load_auto(f"{data_root}/data_s/{dataset}.jsonl")
        out_path = f"{save_dir}/{dataset}.jsonl"
        os.remove(out_path) if osp.exists(out_path) else None

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=get_longbench_pred, args=(rank, world_size, data_subsets[rank], max_length, \
                        max_gen, prompt_format, dataset, device, model_name, model2path, out_path))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    ## Evaluation, borrowed from https://github.com/THUDM/LongBench/blob/main/eval.py
    scores = {}
    for dataset in datasets:
        data_all = load_auto(f"{save_dir}/{dataset}.jsonl")
        predictions = [data['pred'] for data in data_all]
        answers = [data['answers'] for data in data_all]
        all_classes = data_all[0]['all_classes']
        # lengths = [data['length'] for data in data_all]

        score = scorer(dataset, predictions, answers, all_classes)
        scores[dataset] = score
    
    print(f'{model_name} result (saved in {save_dir}/result.json):\n{scores}\n')
    write_auto(f'{save_dir}/result.json', scores)


def eval_attrscore():
    sys.path.append('./third_party/AttrScore')
    from third_party.AttrScore.inference_alpaca import TASK_PROMPTS, confusion_matrix, evaluate_confusion_matrix
    from third_party.AttrScore.train_alpaca import PROMPT_DICT

    evaluator = PreTrainedLLM(f'config/{args.model}.yaml')

    task_prompt, input_template, label_map = TASK_PROMPTS['attribution-with-definition']
    label_regex = r"|".join(list(label_map.keys()))

    if args.use_cot:
        task_prompt += (
            "Take the following steps:\n"
            "1. Read the claim carefully and understand the specific details or measurements mentioned.\n"
            "2. Review the reference and compare the information provided with the details in the claim.\n"
            "3. If there's any discrepancy or lack of specific information in the reference, classify the verification as \"Contradictory\" and \"Extrapolatory\" respectively.\n"
            "4. Double-check your classification before providing an answer to ensure accuracy.\n\n"
        )
    
    prompt_template = PROMPT_DICT['prompt_input']
    model_template = 'llama2'

    data_all = load_auto('data/app/AttrScore/data_to_eval.jsonl')
    for ref_type, ref_key in [('vanilla', 'reference'), ('struct', 'reference_struct')]:
        inputs, labels = [], []
        for data in data_all:
            query, claim, reference, label = \
                data['query'], data['answer'], data[ref_key], data['label']
            labels.append(label)
            
            input = input_template.format(claim, reference)
            inputs.append(prompt_template.format(task_prompt, input))

        predictions: List[str] = evaluator.call_model_local(inputs, batched=True, bs=8, template=model_template) # chat

        true_labels, pred_labels = [], []
        for prediction, true_label in zip(predictions, labels):
            re_matches = re.search(label_regex, prediction, re.IGNORECASE)
            pred_label = re_matches.group().lower() if re_matches is not None else 'None'
            pred_label = label_map[pred_label.capitalize()] if pred_label.capitalize() in label_map else "None"

            true_labels.append(true_label)
            pred_labels.append(pred_label)

        conf_matrix = confusion_matrix(true_labels, pred_labels, labels=["Attributable", "Contradictory", "Extrapolatory"])
        precision, recall, f1, micro_f1, macro_f1 = evaluate_confusion_matrix(conf_matrix)

        # print(conf_matrix)
        # print("Precision:", [round(x*100, 1) for x in precision])
        # print("Recall:", [round(x*100, 1) for x in recall])
        # print("F1:", [round(x*100, 1) for x in f1])
        # print("micro_f1:", micro_f1)
        # print("macro_f1:", macro_f1)
        print(f'{ref_type:10s} | macro_f1: {round(macro_f1, 3)} {[round(y, 3) for y in f1]} ')
    return
    

def eval_factscore():
    data_root = 'data/app/FactScore'
    data_all: Dict[str, List[dict]] = load_auto(f'{data_root}/data_to_eval.json')

    evaluator = PreTrainedLLM(f'config/{args.model}.yaml')

    res_dict = {'vanilla': {}, 'struct': {}}
    for subset, claims in data_all.items():
        for ref_type in res_dict.keys():
            inputs, labels = [], []
            for claim in claims:
                if claim['label'] not in ['NS', 'S']:
                    continue
                labels.append(claim['label'])

                prompt = "Judge a claim True or False based on the given passages.\n\n"
                for pi, psg in enumerate(claim['passages']):
                    key = 'text' if ref_type == 'vanilla' else 'text_struct'
                    prompt += f'Passage {pi+1}: {psg["title"]}\n{psg[key]}\n\n'
                if ref_type == 'vanilla' or True:
                    prompt += f'Now judge: "{claim["text"]}", True or False?'
                else:
                    prompt += f'Now, check information along the structurized passages above and judge: "{claim["text"]}", True or False?'
                # prompt += '\nMake sure your answer is correct and do not output any other words.'
                inputs.append(prompt)

            predictions: List[str] = evaluator.call_model_local(inputs, batched=True, bs=8) # chat
            true_labels, pred_labels = [], []
            for response, label in zip(predictions, labels):
                response = response.lower().replace('true or false', '')
                if "true" in response or "false" in response:
                    if "true" in response and "false" not in response:
                        is_supported = True
                    elif "false" in response and "true" not in response:
                        is_supported = False
                    else:
                        is_supported = response.index("true") > response.index("false")
                else:
                    is_supported = all([keyword not in response.lower().translate(str.maketrans("", "", string.punctuation)).split() for keyword in ["not", "cannot", "unknown", "information"]])
                true_labels.append(label == 'NS')
                # true_labels.append(data['answer'] != 'S')
                pred_labels.append(not is_supported)
            
            true_labels, pred_labels = np.array(true_labels), np.array(pred_labels)
            recall = (true_labels & pred_labels).sum() / true_labels.sum()
            precision = (true_labels & pred_labels).sum() / pred_labels.sum()
            f1score = 2 * recall * precision / (recall + precision + 1e-9)

            res_dict[ref_type][subset] = f1score
    
    for ref_type, results in res_dict.items():
        for subset, f1score in results.items():
            print(f'{ref_type:10s} - {subset:15s} : {f1score*100:.1f}')


def eval_beir():
    raise NotImplementedError(
        'Please run `cd third_party/tevatron && bash scripts/eval_beir.sh MODEL_NAME_OR_PATH`'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default='LongBench', choices=['LongBench', 'AttrScore', 'FactScore', 'BEIR'])
    parser.add_argument('--model', type=str, default='llama2-7b-chat-4k')
    parser.add_argument('--use_cot', action='store_true', default=False)
    args = parser.parse_args()

    eval(f'eval_{args.benchmark.lower()}')()