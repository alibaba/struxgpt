# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import re
from glob import glob
from tqdm import tqdm
from typing import List, Optional, Dict, Union
import argparse
import json
import os.path as osp

import sys
sys.path.append('./')

from src.utils.io import load_auto, write_auto

VERSION = os.environ.get('STRUXGPT', 'v1')
MAX_STRUCT_LEN = 1024
BATCH_SIZE = 8

if VERSION == 'v1':
    from src.models.struxgpt_v1 import StruXGPT, StructItem
else:
    from src.models.struxgpt_v2 import StruXGPT, StructItem

model = StruXGPT()


def preprocess_longbench():
    data_root = 'third_party/LongBench/data'
    temp_root = 'data/app/LongBench'
    save_root = data_root + '_s'
    os.makedirs(temp_root, exist_ok=True)
    os.makedirs(save_root, exist_ok=True)

    def struct_slice(text: str, idx: str, idx2res: Optional[Dict[str, "StructItem"]] = None, 
                     **kwargs) -> Union[Dict[str, str], str]:
        chunk_pairs = model.chunk_content([text], max_length=MAX_STRUCT_LEN, prefix=idx)
        idx2txt = {data['idx']: data['data'] for data in chunk_pairs}

        if idx2res is None:
            return idx2txt
        else:
            struct_list = [idx2res[idx] for idx in idx2txt]
            assert len(struct_list)
            if any(not struct_item.valid for struct_item in struct_list):
                return text
            else:
                if len(struct_list) > 1:
                    scope_list = [struct_item.scope for struct_item in struct_list]
                    final_scope = model('. '.join(scope_list), return_json=False).scope
                    struct_res = StructItem.merge_struct(title=final_scope, struct_list=struct_list)
                else:
                    struct_res = struct_list[0]
                
                hi, fi = kwargs.get('hi', 0), kwargs.get('fi', 1)
                return struct_res.convert_struct_output(hi=hi, fi=fi)

    def struct_slice_multi(text: str, idx: str, idx2res: Optional[Dict[str, "StructItem"]] = None, 
                           sep='Passage (\d+):', ignore=(), **kwargs) -> Dict[str, str]:
        res = {}

        passages_indices = re.findall(sep, text)
        assert len(passages_indices)
        pattern = '|'.join(sep.replace('(\d+)', ind) for ind in passages_indices)
        total_passages = [x for x in re.split(pattern, text)]
        ## first "paragraph" is empty
        assert len(total_passages) == len(passages_indices) + 1, f'{len(total_passages)} != {len(passages_indices)}'
        valid_psg_indices = [i for i, x in enumerate(total_passages) if len(x)]
        assert len(passages_indices) == len(valid_psg_indices)
        assert len(total_passages) == len(valid_psg_indices) + 1, f'{len(total_passages)} != {len(valid_psg_indices)}'
        passages = [total_passages[i] for i in valid_psg_indices]
        assert len(passages) > 1
        
        for pi, passage in enumerate(passages):
            if len(ignore):
                passage_filtered = re.split('|'.join(ignore), passage)[0]
            else:
                passage_filtered = passage

            if idx2res is None:
                res.update(struct_slice(passage_filtered, f'{idx}_{pi}'), **kwargs)
            else:
                struct_res = struct_slice(passage_filtered, f'{idx}_{pi}', idx2res, **kwargs)
                prefix = sep.replace('(\d+)', passages_indices[pi])
                assert prefix + passage in text
                text = text.replace(prefix + passage, prefix + passage.replace(passage_filtered, struct_res))
        
        if idx2res is not None:
            res = text

        return res

    ds2func = {
        'qasper': (struct_slice, {}),
        'multifieldqa_en': (struct_slice, {}),
        'hotpotqa': (struct_slice_multi, {'sep': 'Passage (\d+):\n'}),
        '2wikimqa': (struct_slice_multi, {'sep': 'Passage (\d+):\n'}),
        'musique': (struct_slice_multi, {'sep': 'Passage (\d+):\n'}),
        'passage_count': (struct_slice_multi, {'sep': 'Paragraph (\d+): '}),
        'passage_retrieval_en': (struct_slice_multi, {'sep': 'Paragraph (\d+): '}),
    }
    struct_res_total = []
    for ds, (chunk_func, kwargs) in ds2func.items():
        data_all = load_auto(f'{data_root}/{ds}.jsonl')
        for i, data in enumerate(tqdm(data_all, desc=ds)):
            ctx_key = 'context'
            idx2data: Dict[str, str] = chunk_func(data[ctx_key], f'{ds}_{i}', **kwargs)
            indices, text_to_struct = zip(*list(idx2data.items()))
            struct_results = model.batch_forward(text_to_struct, return_json=False, bs=BATCH_SIZE, show_progress=False)
            idx2res = {idx: struct_item for idx, struct_item in zip(indices, struct_results)}
            structurized_context = chunk_func(data[ctx_key], f'{ds}_{i}', idx2res, **kwargs)
            data[ctx_key] = structurized_context
            struct_res_total.extend([{'idx': idx, 'struct_res': struct_item.dict} \
                                        for idx, struct_item in idx2res.items()])
        write_auto(f'{save_root}/{ds}.jsonl', data_all)
    write_auto(f'{temp_root}/struct_res.jsonl', struct_res_total)
    

def preprocess_attrscore():
    from datasets import load_dataset

    data_root = './third_party/AttrScore/AttrScore'
    save_root = 'data/app/AttrScore'
    os.makedirs(save_root, exist_ok=True)

    subset_name = 'attreval_gensearch'
    data_all = [row for row in load_dataset(data_root, subset_name)['test']]

    references = [data['reference'] for data in data_all]
    assert max(model.count_token(ref) for ref in references) < MAX_STRUCT_LEN * 1.5
    struct_results = model.batch_forward(references, return_json=False, bs=BATCH_SIZE, show_progress=True)

    # TODO: temporal
    # struct_results = [{'idx': f'attr_{i}', 'struct_res': struct_item.dict} \
    #                         for i, struct_item in enumerate(struct_results)]
    # write_auto(f'{save_root}/struct_res.jsonl', struct_results)

    # struct_results = load_auto(f'{save_root}/struct_res.jsonl')
    # struct_results = [StructItem(struct_dict=data['struct_res']) for data in struct_results]

    for (data, struct_res) in zip(data_all, struct_results):
        if struct_res.valid:
            # TODO: optional
            # model.check_struct_missing_desc(struct_res)
            # model.remapping_struct_source(struct_res)
            data['reference_struct'] = struct_res.convert_struct_output(hi=0, fi=1)
        else:
            data['reference_struct'] = data['reference']

    write_auto(f'{save_root}/data_to_eval.jsonl', data_all)


def preprocess_factscore():
    factscore_root = './third_party/FActScore'
    sys.path.append(factscore_root)
    from factscore.factscorer import FactScorer

    data_root = 'data/app/FactScore'
    os.makedirs(data_root, exist_ok=True)

    ## Get data offline
    cache_dir = f'{factscore_root}/.cache/factscore/'
    fs = FactScorer(model_name='',
                    data_dir=cache_dir,
                    model_dir=cache_dir,
                    cache_dir=cache_dir,)

    subset2claims = {}
    for file_path in sorted(glob(f'{factscore_root}/data/labeled/*.jsonl')):
        subset = osp.basename(file_path).split('.')[0]

        topics, atomic_facts_with_human_labels = [], []
        for dp in load_auto(file_path):
            if dp["annotations"] is None:
                continue
            topics.append(dp["topic"])

            facts_tmp = []
            for sent in dp['annotations']:
                if sent['human-atomic-facts'] is not None:
                    facts_tmp.extend(sent['human-atomic-facts'])
            atomic_facts_with_human_labels.append(facts_tmp)
                
        claims = fs.get_passages(topics=topics,
                                 atomic_facts=atomic_facts_with_human_labels,
                                 verbose=True)
        # write_auto(f'{data_root}/{subset}.jsonl', claims)

        # claims = load_auto(f'{data_root}/{subset}.jsonl')

        subset2claims[subset] = claims

    total_passages = []
    for claims in subset2claims.values():
        for claim in claims:
            for psg in claim['passages']:
                total_passages.append(psg['text'])
    total_passages = list(set(total_passages))

    assert max(model.count_token(ref) for ref in total_passages) < MAX_STRUCT_LEN * 1.5
    struct_results = model.batch_forward(total_passages, return_json=False, bs=BATCH_SIZE, show_progress=True)
    # # TODO: optional
    # for struct_item in struct_results:
    #     if struct_item.valid:
    #         model.check_struct_missing_desc(struct_item)
    #         model.remapping_struct_source(struct_item)

    # TODO: temporal
    # struct_results = [{'idx': f'fact_{i}', 'struct_res': struct_item.dict} \
    #                         for i, struct_item in enumerate(struct_results)]
    # write_auto(f'{data_root}/struct_res.jsonl', struct_results)

    # struct_results = load_auto(f'{data_root}/struct_res.jsonl')
    # struct_results = [StructItem(struct_dict=data['struct_res']) for data in struct_results]

    psg2struct = {psg: struct_item for psg, struct_item in zip(total_passages, struct_results)}

    for subset, claims in subset2claims.items():
        for claim in claims:
            for psg in claim['passages']:
                struct_res = psg2struct[psg['text']]
                if struct_res.valid:
                    psg['text_struct'] = struct_res.convert_struct_output(hi=0, fi=1)
                else:
                    psg['text_struct'] = psg['text']

    write_auto(f'{data_root}/data_to_eval.json', subset2claims)


def preprocess_beir():
    from datasets import load_dataset
    SPLITER = '_'

    data_root = 'third_party/tevatron/datasets/beir-corpus-struct'
    # beir_path = 'data/app/BEIR'
    beir_path = 'third_party/tevatron/datasets'
    data_dict: Dict[str, List[dict]] = {}
    data_to_infer = {}
    for subset in ['nfcorpus', 'fiqa', 'arguana', 'scidocs', 'scifact']:
        dataset = load_dataset(f'{beir_path}/beir-corpus', subset)['train']
        data_dict[subset] = []
        for di, data in enumerate(dataset):
            data_dict[subset].append(data)
            passage: str = data['text']
            if not (50 < len(passage.split()) < 1024):
                continue
            data_to_infer[f'{subset}{SPLITER}{di}'] = passage
    
    total_passages = list(data_to_infer.values())
    struct_results = model.batch_forward(total_passages, return_json=False, bs=BATCH_SIZE, show_progress=True)
    # # TODO: optional
    # for struct_item in struct_results:
    #     if struct_item.valid:
    #         model.check_struct_missing_desc(struct_item)
    #         model.remapping_struct_source(struct_item)

    # TODO: temporal
    # struct_results = [{'idx': f'fact_{i}', 'struct_res': struct_item.dict} \
    #                         for i, struct_item in enumerate(struct_results)]
    # write_auto(f'{data_root}/struct_res.jsonl', struct_results)

    # struct_results = load_auto(f'{data_root}/struct_res.jsonl')
    # struct_results = [StructItem(struct_dict=data['struct_res']) for data in struct_results]

    for (idx, passage), struct_res in zip(data_to_infer, struct_results):
        if not struct_res.valid:
            continue
        *subset, di = idx.split(SPLITER)
        subset = SPLITER.join(subset)
        di = int(di)
        data_dict[subset][di]['text'] = struct_res.convert_struct_output(hi=0, fi=2)

    for subset, data_all in data_dict.items():
        write_auto(f'{data_root}/{subset}.jsonl', data_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default='BEIR', choices=['LongBench', 'AttrScore', 'FactScore', 'BEIR'])
    args = parser.parse_args()

    eval(f'preprocess_{args.benchmark.lower()}')()
