# Copyright (c) Alibaba, Inc. and its affiliates.
import random
import argparse
import os
import os.path as osp
from typing import List, Dict, Union

import sys
sys.path.append('./')

from src.utils import load_auto, write_auto
from src.models.struxgpt_base import PreTrainedLLM

sys.path.append(osp.dirname(__file__))
from utils import parse_answer

BATCH_SIZE = 32


def eval_longbench(args):
    sys.path.append('./third_party/LongBench')
    from third_party.LongBench.pred import seed_everything
    from third_party.LongBench.eval import scorer

    def _get_items(data_all: List[dict], key):
        return [x[key] for x in data_all]

    seed_everything(42)

    evaluator = PreTrainedLLM(args.model)

    save_root = 'third_party/LongBench'
    model_name = osp.basename(evaluator.model_name_or_path)
    save_dir = f"{save_root}/pred_cb/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    qa_eval_all: Dict[str, List[Dict[str, str]]] = load_auto('data/tune/LongBench/qa_eval.json')
    scores = {}
    for subset, qa_list in qa_eval_all.items():
        questions = _get_items(qa_list, 'question')
        predictions = evaluator(questions, batched=True, bs=BATCH_SIZE, progress_desc=subset)
        predictions = [parse_answer(pred) for pred in predictions]
        answers = _get_items(qa_list, 'answers')
        all_classes = _get_items(qa_list, 'all_classes')

        score = scorer(subset, predictions, answers, all_classes)
        scores[subset] = score
    
    save_path = f'{save_dir}/result.json'
    print(f'{model_name}\'s result (saved in {save_path}):\n{scores}\n')
    write_auto(save_path, scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='config/llama2_7b.yaml')
    args = parser.parse_args()

    eval_longbench(args)