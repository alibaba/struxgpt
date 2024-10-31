# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import re
from glob import glob
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from typing import List, Dict, Union, Literal, Tuple
import argparse
import json
import pdb
import os.path as osp
import shutil
import random
random.seed(1234)
np.random.seed(1234)

import sys
sys.path.append('./')

from src.utils import load_auto
from src.models.struxgpt_base import PreTrainedLLM

sys.path.append(osp.dirname(__file__))
from utils import format_trainval_qa, eval_accuracy

BATCH_SIZE = 8


def mmedbench_build_test_data(data_dir):
    data_dict_all: Dict[str, Tuple[List[str]]] = {}
    for path in sorted(glob(f'{data_dir}/*jsonl')):
        lang = osp.basename(path).split('.')[0]
        data_dict_all[lang] = ([], [])
        for src_dict in load_auto(path):
            # TODO: temporal solution
            prompt, answer_id, _, _ = format_trainval_qa(lang, src_dict, is_with_rationale=False, 
                                                         use_alpaca=False, use_cot=True)
            prompt = prompt.replace(' Think step by step.', '')
            data_dict_all[lang][0].append(prompt)
            data_dict_all[lang][1].append(answer_id)
    
    return data_dict_all


def mmedbench_parse_answer(answer: str):
    if answer.startswith('###'):
        pattern = "OPTION (.*) IS CORRECT"
        # answer = answer.split('OPTION ')[1].split()[0]
    else:
        pattern = "(.*). "

    res = re.findall(pattern, answer)
    return res[0] if len(res) else answer


def eval_mmedbench(args):
    evaluator = PreTrainedLLM(args.model)
    data_dir = args.data_dir

    test_data = mmedbench_build_test_data(data_dir)

    model_name = osp.basename(evaluator.model_name_or_path)
    acc_dict = {}
    for lang, (questions, answers) in test_data.items():
        assert len(questions) == len(answers)
        predictions = evaluator(questions, batched=True, bs=BATCH_SIZE, progress_desc=lang)
        assert len(predictions) == len(answers)
        accuracy = [eval_accuracy(mmedbench_parse_answer(pred), 
                                  mmedbench_parse_answer(ans)) for pred, ans in zip(predictions, answers)]
        acc = np.mean(accuracy)
        acc_dict[lang] = acc

    s = f'{model_name}:\n'
    for lang, acc in acc_dict.items():
        s += f'{lang}={acc*100:.2f}%; '
    s += f'Average={np.mean(list(acc_dict.values()))*100.:.2f}%.'

    print(s)
    with open('./eval.log', 'a') as f:
        f.write(s + '\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='config/llama2_7b.yaml')
    parser.add_argument('--data_dir', type=str, default='data/tune/MMedBench/MMedBench/MMedBench/Test')
    args = parser.parse_args()

    eval_mmedbench(args)