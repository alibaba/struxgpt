# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import re
from glob import glob
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from typing import List, Optional, Dict, Union, Literal
import argparse
import json
import os.path as osp
import random
random.seed(1234)
np.random.seed(1234)

import sys
sys.path.append('./')

from src.utils import (
    load_auto, write_auto,
    drop_last_segment, remove_punc, upper_start, is_zh,
    is_empty, white_space_fix,
    f1_score_auto
)
from src.models.struxgpt_base import PreTrainedLLM
from src.models.struxgpt_v2 import StruXGPT, StructItem

sys.path.append(osp.dirname(__file__))
from utils import format_qa, format_answer, format_question


MAX_STRUCT_LEN = 1024
BATCH_SIZE = 16
GENERATOR_LLM = 'config/llama2_7b.yaml'

model = StruXGPT(debug=False)


################ LongBench ################
longbench_subsets = ['qasper', 'multifieldqa_en', 'multifieldqa_zh',
                     'hotpotqa', '2wikimqa', 'musique', 'dureader']

def preprocess_longbench(
    data_root='third_party/LongBench/data',
    save_root='data/tune/LongBench',
    train_root='third_party/LLaMA-Factory/data/longbench'
):
    os.makedirs(save_root, exist_ok=True)

    psg_chunk_path = f'{save_root}/longbench_psg_chunks_all.jsonl'
    if not osp.exists(psg_chunk_path):
        longbench_extract_passages(data_root, psg_chunk_path)
        longbench_struct_passages(psg_chunk_path)

    psg_struct_path = f'{save_root}/longbench_psg_structs.json'
    if not osp.exists(psg_struct_path):
        longbench_reconst_passages(psg_chunk_path, psg_struct_path)
    
    os.makedirs(train_root, exist_ok=True)
    cpt_train_path = f'{train_root}/longbench_psg_train_cpt.json'
    if not osp.exists(cpt_train_path):
        longbench_gen_struct_train(data_root, psg_chunk_path, psg_struct_path, 
                                   cpt_train_path, mode='cpt')
    
    sft_train_path = f'{train_root}/longbench_psg_train_sft.json'
    if not osp.exists(sft_train_path):
        longbench_gen_struct_train(data_root, psg_chunk_path, psg_struct_path, 
                                   sft_train_path, mode='sft')
    
    print('Done.')


def longbench_extract_passages(data_root, psg_chunk_path):
    def struct_slice(text: str, idx: str, max_token=MAX_STRUCT_LEN, **kwargs):
        res = []

        def _add_res(idx, content, res_list: List[Dict[str, str]]):
            res_list.append({'idx': idx, 'content': content})

        prev_paras, prev_tokens = [], 0
        for para in text.splitlines():
            token = model.count_token(para)
            if (token + prev_tokens > max_token) \
                    and len(prev_paras) and len(prev_paras[-1]) and prev_paras[-1][-1] in '.?!。？！':
                _add_res(f'{idx}_{len(res)}', '\n'.join(prev_paras), res)
                prev_paras, prev_tokens = [], 0
            
            prev_paras.append(para)
            prev_tokens += token

        if any(len(para.split()) for para in prev_paras):
            _add_res(f'{idx}_{len(res)}', '\n'.join(prev_paras), res)

        return res

    def struct_slice_multi(text: str, idx: str, sep='Passage (\d+):', ignore=(), **kwargs):
        res = []

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

            res.extend(struct_slice(passage_filtered, f'{idx}_{pi}'), **kwargs)
        
        return res

    ds2func = {
        'qasper': (struct_slice, {}),
        'multifieldqa_en': (struct_slice, {}),
        'multifieldqa_zh': (struct_slice, {}),
        'hotpotqa': (struct_slice_multi, {'sep': 'Passage (\d+):\n'}),
        '2wikimqa': (struct_slice_multi, {'sep': 'Passage (\d+):\n'}),
        'musique': (struct_slice_multi, {'sep': 'Passage (\d+):\n'}),
        'dureader': (struct_slice_multi, {'sep': '文章(\d+)\n'}),  # 标题：
    }
    print('Extracting LongBench passages')
    passages_total = []
    for ds, (chunk_func, kwargs) in ds2func.items():
        data_all = load_auto(f'{data_root}/{ds}.jsonl')
        for i, data in enumerate(tqdm(data_all, desc=ds)):
            ctx_key = 'context'
            idx2data: List[Dict[str, str]] = chunk_func(data[ctx_key], f'{ds}_{i}', **kwargs)
            passages_total.extend(idx2data)

    write_auto(psg_chunk_path, passages_total, disp=True)


def longbench_struct_passages(psg_chunk_path):
    print('Structurizing passage chunks')
    psg_chunk_all = load_auto(psg_chunk_path)
    if psg_chunk_all[0].get('raw_response', None):
        return
    psg_chunk_list = [x['content'] for x in psg_chunk_all]
    psg_chunk_struct_res = model.batch_forward(psg_chunk_list, return_json=False, bs=BATCH_SIZE, show_progress=True)
    for data, res in zip(psg_chunk_all, psg_chunk_struct_res):
        data['raw_query'] = res.raw_query
        data['raw_response'] = res.raw_response
    write_auto(psg_chunk_path, psg_chunk_all)


def longbench_reconst_passages(psg_chunk_path, psg_struct_path):
    psg_chunk_all = load_auto(psg_chunk_path)
    generator = PreTrainedLLM(GENERATOR_LLM)

    summary_template = (
        "Summarize the following content with NO MORE THAN EIGHT words:\n"
        "```\n{content}\n```\n\n"
        "Your answer should use the same language (English or 中文) as the content, and follow this format:\n"
        "```[the summary goes here]```"
        "\n\nDo not output any other words."
    )

    def _parse_scope(scope: str):
        try:
            sentences = [line for line in scope.splitlines() \
                            if not is_empty(line) and model.count_token(line) > 4]
            if upper_start(white_space_fix(sentences[0]), False).startswith('sure') \
                    or 'summar' in sentences[0]:
                if white_space_fix(sentences[0]).endswith(':'):
                    scope = sentences[1]
                else:
                    scope = sentences[0].split(': ')[1]
            else:
                scope = sentences[0]
            # scope = sentences[1] if 'summary' in sentences[0] else sentences[0]
            scope = scope.split('"')[1] if '"' in scope else scope
            scope = scope.split('[')[1].split(']')[0] if '[' in scope else scope
            scope = scope.split('```')[1] if '`' in scope else scope
        except:
            pass
        if len(scope) and not scope[0] in '1234567890':
            scope = model.split_to_sentence(scope)[0]
        scope = remove_punc(scope)
        # print(scope)
        return scope

    def _check_len(content, max_tokens=MAX_STRUCT_LEN * 3):
        if model.count_token(content) > max_tokens:
            sentences = model.split_to_sentence(content)
            tokens = [model.count_token(sent) for sent in sentences]
            if sum(tokens) > max_tokens:
                num = np.nonzero(np.cumsum(tokens) > max_tokens)[0][0]
                sentences = sentences[:num]
            content = ' '.join(sentences)
        return content

    psg2res: Dict[str, Dict[str, list]] = {}
    psg_to_sum: List[List[str]] = []
    for chunk in tqdm(psg_chunk_all, desc='Merge Chunks'):
        psg_idx, chunk_ind = drop_last_segment(chunk['idx'], last_mapping=int)
        if psg_idx not in psg2res:
            psg2res[psg_idx] = {'chunks': []}
        assert chunk_ind == len(psg2res[psg_idx]['chunks'])
        struct_res = StructItem(model.scfg, chunk['raw_query'], chunk['raw_response'])
        if not struct_res.valid:
            content = _check_len(chunk['content'])
            psg_to_sum.append([psg_idx, chunk_ind, summary_template.format(content=content)])
            # scope = generator(summary_template.format(content=content))
            # struct_res.scope = _parse_scope(scope)
        psg2res[psg_idx]['chunks'].append(struct_res.to_json())
    psg_scopes = generator([x[-1] for x in psg_to_sum], batched=True, bs=BATCH_SIZE, 
                             progress_desc='Extracte Chunk Title')
    assert len(psg_to_sum) == len(psg_scopes), f'{len(psg_to_sum)} {len(psg_scopes)}'
    for (psg_idx, chunk_ind, _), scope in zip(psg_to_sum, psg_scopes):
        psg2res[psg_idx]['chunks'][chunk_ind]['scope'] = _parse_scope(scope)

    psg_to_struct = []
    for psg_idx, psg_struct in psg2res.items():
        scope_list = [x['scope'] for x in psg_struct['chunks']]
        # scope_total = model.mapping_sentence(scope_list)
        scope_total = '. '.join(scope_list)
        psg_to_struct.append(summary_template.format(content=scope_total))
    psg_titles = generator(psg_to_struct, batched=True, bs=BATCH_SIZE, 
                             progress_desc='Extract Passage Title')
    psg_titles = [_parse_scope(x) for x in psg_titles]

    total_psg_indices = list(psg2res.keys())
    assert len(total_psg_indices) == len(psg_titles)
    for psg_idx, psg_title in zip(total_psg_indices, psg_titles):
        psg2res[psg_idx]['extracted_title'] = psg_title

    multidoc_psg: Dict[str, Dict[str, List[str]]] = {}
    for psg_idx, psg_struct in psg2res.items():
        if re.search('_(\d+)_(\d+)', psg_idx) is None:
            continue
        sample_idx, psg_ind = drop_last_segment(psg_idx, last_mapping=int)
        if sample_idx not in multidoc_psg:
            multidoc_psg[sample_idx] = {'psg_idx': [], 'psg_title': []}
        assert psg_ind == len(multidoc_psg[sample_idx]['psg_idx'])
        multidoc_psg[sample_idx]['psg_idx'].append(psg_idx)
        multidoc_psg[sample_idx]['psg_title'].append(psg_struct['extracted_title'])
    
    article_to_struct = []
    for sample_idx, sample_struct in multidoc_psg.items():
        # scope_total = model.mapping_sentence(sample_struct['psg_title'])
        scope_total = '. '.join(sample_struct['psg_title'])
        scope_total = _check_len(scope_total)
        article_to_struct.append(summary_template.format(content=scope_total))
    article_titles = generator(article_to_struct, batched=True, bs=BATCH_SIZE, 
                                 progress_desc='Extract MultiDoc Title')
    article_titles = [_parse_scope(x) for x in article_titles]

    total_article_indices = list(multidoc_psg.keys())
    assert len(total_article_indices) == len(article_titles)
    for article_idx, article_title in zip(total_article_indices, article_titles):
        multidoc_psg[article_idx]['total_title'] = article_title

    assert len(set(total_psg_indices) & set(total_article_indices)) == 0

    psg2res.update(multidoc_psg)

    write_auto(psg_struct_path, psg2res)


def longbench_gen_struct_train(data_root,
                               psg_chunk_path, psg_struct_path, train_data_path, 
                               mode: Literal['cpt', 'sft']):
    psg2res: Dict[str, Dict[str, List[str]]] = load_auto(psg_struct_path)
    psg_chunk_all: List[Dict[str, str]] = load_auto(psg_chunk_path)
    idx2chunk: Dict[str, str] = {chunk['idx'] : chunk for chunk in psg_chunk_all}
    if mode == 'sft':
        ref_qa_dict = longbench_load_qa_data(data_root)
        qa_gen_template = load_auto('src/templates/qa_gen/longbench_ssft_with_demos.md')
        ds2prompt_close = load_auto(f'{osp.dirname(data_root)}/config/dataset2prompt_close.json')
        ds2prompt_close_cot = load_auto(f'{osp.dirname(data_root)}/config/dataset2prompt_close_cot.json')
        qa_gen_data_all = []

    vanilla_train_data_all: List[Dict[str, str]] = []
    struct_train_data_all: List[Dict[str, str]] = []
    for psg_idx, psg_struct in tqdm(psg2res.items(), desc='Parse Passages', total=len(psg2res)):
        multidoc_abstract = psg_struct.get('psg_idx', None) is not None
        if multidoc_abstract:
            psg_indices = psg_struct['psg_idx']
            ctx_title = psg_struct['total_title']
        elif re.search('_(\d+)_(\d+)', psg_idx) is not None:
            # multi doc's passage, remove dulplication
            continue
        else:
            psg_indices = [psg_idx]
            ctx_title = psg_struct['extracted_title']
        
        psg_structs, psg_chunk_structs, psg_contents = [], [],[]
        for _psg_idx in psg_indices:
            psg_title = psg2res[_psg_idx]['extracted_title']
            psg_chunks = psg2res[_psg_idx]['chunks']
            chunk_struct_list, chunk_content_list = [], []
            for chunk_ind, chunk in enumerate(psg_chunks):
                struct_item = StructItem(model.scfg, chunk['raw_query'], chunk['raw_response'])
                if not struct_item.valid:
                    struct_item = StructItem.lazzy_struct(chunk['scope'], 
                                                          model.remapping_sentence( chunk['raw_query']))
                chunk_struct_list.append(struct_item)
                chunk_idx = f'{_psg_idx}_{chunk_ind}'
                assert idx2chunk[chunk_idx]['raw_query'] == chunk['raw_query']
                chunk_content_list.append(idx2chunk[chunk_idx]['content'])
            psg_struct = StructItem.merge_struct(psg_title, chunk_struct_list, 'upgrade').to_json()
            StructItem.prune_struct_level(psg_struct, preserve_level=1)
            psg_structs.append(psg_struct)
            psg_chunk_structs.append(chunk_struct_list)
            psg_contents.append(chunk_content_list)
        
        if multidoc_abstract:
            assert len(psg_structs) > 1
            ctx_struct = StructItem.merge_struct(ctx_title, psg_structs, 'upgrade').to_json()
            StructItem.prune_struct_level(ctx_struct, preserve_level=-1)
        else:
            assert len(psg_structs) == 1
            ctx_struct = psg_structs[0]
        ctx_struct = StructItem(struct_dict=ctx_struct)
        ctx_idx = psg_idx
    
        if mode == 'cpt':
            for psg_ind, (psg_struct, chunk_struct_list, chunk_content_list) in \
                    enumerate(zip(psg_structs, psg_chunk_structs, psg_contents)):
                ## Vanilla CPT
                vanilla_train_data_all.append({
                    'text': '\n'.join(chunk_content_list)
                })

                ## Struct CPT
                assert len(chunk_struct_list) == len(chunk_content_list)
                for chunk_ind, (chunk_struct, chunk_content) in \
                        enumerate(zip(chunk_struct_list, chunk_content_list)):
                    chunk_struct.scope = remove_punc(chunk_struct.scope)
                    struct_train_data_all.append(
                        model.gen_struct_corpus(ctx_struct, chunk_struct, chunk_content, f'{ctx_idx}_{psg_ind}_{chunk_ind}')
                    )
            struct_train_data_all.append(
                model.gen_mindmap_corpus(ctx_struct, f'{ctx_idx}_mindmap')
            )
        elif mode == 'sft':
            subset, ind = drop_last_segment(ctx_idx, last_mapping=int)
            ref_qa = ref_qa_dict[subset][ind]
            demos = format_qa(ref_qa)
            
            chunk_name_total, chunk_content_total = [], []
            for chunk_struct_list, chunk_content_list in zip(psg_chunk_structs, psg_contents):
                chunk_name_total.extend([remove_punc(chunk_struct.scope) for chunk_struct in chunk_struct_list])
                chunk_content_total.extend(chunk_content_list)
            
            for _ in range(2):
                total_chunk_num = len(chunk_content_total)
                chunk_indices = list(range(total_chunk_num))
                chunk_num_range = [1, 3] if not multidoc_abstract else [2, min(4, total_chunk_num+1)]

                random.shuffle(chunk_indices)
                chosen_chunk_num = np.random.randint(*chunk_num_range)
                chosen_chunk_ids = sorted(chunk_indices[:chosen_chunk_num])
                chosen_titles   = [chunk_name_total[i] for i in chosen_chunk_ids]
                chosen_contents = [chunk_content_total[i] for i in chosen_chunk_ids]

                qa_gen_data = model.gen_struct_qa(qa_gen_template, ctx_struct, chosen_titles, chosen_contents, 
                                                  f'{ctx_idx}', demonstrations=demos, prompt_only=True)
                qa_gen_data.update({'ref_qa': deepcopy(ref_qa), 'subset': subset})
                qa_gen_data_all.append(qa_gen_data)

            ## Add title for QA data
            longbench_reconst_qa_data(ref_qa, ctx_title)
            ref_qa['question'] = format_question(ref_qa['question'], ds2prompt_close[subset])
            # ref_qa['answer'] = format_answer(ref_qa['answer'])

    if mode == 'sft':
        save_root = osp.dirname(psg_chunk_path)
        write_auto(f'{save_root}/qa_eval.json', ref_qa_dict)

        generator = PreTrainedLLM(GENERATOR_LLM)
        qa_prompts = [x['question'] for x in qa_gen_data_all]
        qa_responses = generator(qa_prompts, batched=True, bs=BATCH_SIZE, 
                                 progress_desc='Generating QA Samples')
        assert len(qa_gen_data_all) == len(qa_responses) == len(qa_prompts)
        
        for qa_gen_data, qa_gen_response in zip(qa_gen_data_all, qa_responses):
            ctx_struct: "StructItem" = qa_gen_data['ctx_struct']
            ctx_mindmap: str = qa_gen_data['ctx_mindmap']
            idx, subset = qa_gen_data['idx'], qa_gen_data['subset']
            ref_qa = qa_gen_data['ref_qa']
            qa_zh = is_zh(ref_qa['question'])

            qa_data = model.gen_struct_qa(qa_gen_template, ctx_struct, [], [], idx, 
                                          response=qa_gen_response, parse_only=True)
            if len(qa_data) and f1_score_auto(qa_data['question'], ref_qa['question']) < 0.7:
                longbench_reconst_qa_data(qa_data, ctx_struct.scope)

                question, answer = qa_data['question'], qa_data['answer']
                explanation = qa_data['explanation']

                ## Vanilla SFT
                vanilla_train_data_all.append({
                    'prompt': format_question(question, ds2prompt_close[subset]),
                    'response': format_answer(answer)
                })
                # vanilla_train_data_all.append({
                #     'prompt': format_question(question, ds2prompt_close_cot[subset]),
                #     'response': format_answer(answer, explanation)
                # })

                ## Struct SFT
                if qa_zh:
                    mindmap = f'{ctx_struct.scope}的知识结构如下：\n{ctx_mindmap}\n\n'
                else:
                    mindmap = f'Here is the knowledge structure regarding {ctx_struct.scope}:\n{ctx_mindmap}\n\n'
                explanation = mindmap + explanation
                struct_train_data_all.append({
                    'prompt': format_question(question, ds2prompt_close[subset]),
                    'response': format_answer(answer)
                })
                struct_train_data_all.append({
                    'prompt': format_question(question, ds2prompt_close_cot[subset]),
                    'response': format_answer(answer, explanation)
                })
            
            if qa_zh:
                mindmap_question = f'{ctx_struct.scope}的知识结构是什么？'
            else:
                mindmap_question = f'What is the knowledge structure regarding {ctx_struct.scope}?'
            mindmap_answer = ctx_mindmap.replace("**", "")
            struct_train_data_all.append({
                'prompt': mindmap_question,
                'response': mindmap_answer
            })

    vanilla_train_data_path = train_data_path
    write_auto(vanilla_train_data_path, vanilla_train_data_all)
    struct_train_data_path = train_data_path.replace(mode, 's' + mode)
    write_auto(struct_train_data_path, struct_train_data_all)


def longbench_load_qa_data(data_root):
    data_dict: Dict[str, List[Dict[str, str]]] = {}
    for subset in longbench_subsets:
        data_all = load_auto(f'{data_root}/{subset}.jsonl')
        data_dict[subset] = []
        for item in data_all:
            data_dict[subset].append({
                'question': item['input'],
                'answer': item['answers'][0],
                'answers': item['answers'],
                'all_classes': item['all_classes']
            })

    return data_dict


def longbench_reconst_qa_data(qa_data: Dict[str, str], title: str):
    question = qa_data['question']
    if is_zh(question):
        if not question.startswith(f'关于{title}，'):
            question = f"关于{title}，{question}" 
    else:
        # title = upper_start(title, False)
        if not question.startswith(f'Regarding {title}, '):
            question = upper_start(question, False)
            question = f"Regarding {title}, {question}"
    qa_data['question'] = question


if __name__ == '__main__':
    preprocess_longbench()