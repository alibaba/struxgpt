# Copyright (c) Alibaba, Inc. and its affiliates.
from tqdm import tqdm
import re
import argparse

import sys
sys.path.append('./')

from src.utils.io import load_auto, write_auto


def prepare_data(version='v1'):
    if version == 'v1':
        from src.models.struxgpt_v1 import StruXGPT, StructItem
    else:
        from src.models.struxgpt_v2 import StruXGPT, StructItem
    model = StruXGPT(debug=True)

    data_src = 'data/tune/StruXGPT'
    data_dst = 'third_party/LLaMA-Factory/data/struxgpt'
    for split in ['train', 'val']:
        data_all = load_auto(f'{data_src}/struxgpt_{version}_{split}_raw.jsonl')
        data_new = []
        for struct_dict in tqdm(data_all, desc=split):
            struct_item = StructItem(struct_dict=struct_dict)
            assert struct_item.valid

            input = struct_item.raw_query
            kwargs = {'input_sent_enum': True} if version == 'v2' else {}
            prompt = model.prepare_prompt(input, **kwargs)
            response = str(struct_item)
            assert StructItem(model.scfg, input, response).valid

            system = model.prompt_system
            data_new.append({'system': system, 'prompt': prompt, 'response': response})

        write_auto(f'{data_dst}/{version}_{split}.json', data_new)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='v1', choices=['v1', 'v2'])
    args = parser.parse_args()

    prepare_data(args.version)