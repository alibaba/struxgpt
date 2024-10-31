# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import re
from glob import glob
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from typing import List, Dict, Union, Literal, Tuple
from collections import Counter
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

from src.utils import (
    load_auto, write_auto,
    drop_last_segment, remove_punc, upper_start, is_zh, PUNCTUATION,
    is_empty, white_space_fix,
    f1_score_auto
)
from src.models.struxgpt_base import PreTrainedLLM
from src.models.struxgpt_v2 import StruXGPT, StructItem

sys.path.append(osp.dirname(__file__))
from rag import LlamaIndexRetriever
from utils import format_demo_qa, format_trainval_qa

MAX_STRUCT_LEN = 1024
MAX_CORPUS_LEN = 3072
BATCH_SIZE = 8
GENERATOR_LLM = 'config/llama2_7b.yaml'

model = StruXGPT(debug=True)


################ MMedBench ################
subsets = ['English', 'Chinese', 'French', 'Japanese', 'Russian', 'Spanish']

def preprocess_mmedbench(
    data_root='data/tune/MMedBench',
    train_root='third_party/LLaMA-Factory/data/mmedbench'
):
    os.makedirs(data_root, exist_ok=True)

    corpus_chunk_path = f'{data_root}/mmedbench_corpus_chunks_all.json'
    if not osp.exists(corpus_chunk_path):
        mmedbench_gather_corpus(data_root, corpus_chunk_path)
        mmedbench_summ_chunks(corpus_chunk_path)

    corpus_struct_path = f'{data_root}/mmedbench_corpus_structs.json'
    if not osp.exists(corpus_struct_path):
        mmedbench_struct_corpus(corpus_chunk_path, corpus_struct_path)
    
    os.makedirs(train_root, exist_ok=True)
    cpt_train_path = f'{train_root}/mmedbench_corpus_train_cpt.json'
    if not osp.exists(cpt_train_path):
        mmedbench_gen_struct_train(data_root, corpus_struct_path, cpt_train_path, 
                                   mode='cpt')
    
    sft_train_path = f'{train_root}/mmedbench_instruct_train_sft.json'
    if not osp.exists(sft_train_path):
        mmedbench_gen_struct_train(data_root, corpus_struct_path, cpt_train_path, 
                                   mode='sft')
    
    print('Done.')


def mmedbench_gather_corpus(data_root, corpus_chunk_path):
    data_dict_all: Dict[str, List[Dict[str, Union[str, List[Dict[str, str]]]]]] = {}

    ### English & Chinese from textbooks
    textbook_cfg = {'English': 'en', 'Chinese': 'zh_paragraph'}
    for lang, subdir in textbook_cfg.items():
        data_dict_all[lang] = []
        for src_path in tqdm(sorted(glob(f'{data_root}/textbooks/{subdir}/*.txt')), 
                             desc=f'Collecting {lang}'):
            if 'all_books' in src_path:
                continue

            bookname = osp.basename(src_path).split('.')[0]
            if '_' in bookname and 'First_Aid' not in bookname:
                bookname = drop_last_segment(bookname)[0]

            content = load_auto(src_path)

            data_dict_all[lang].append({
                'bookname': bookname, 
                'content': content
            })

    ### French & Japanese (empty) & Russian & Spanish from MMedC
    languages = ['French', 'Japanese', 'Russian', 'Spanish']
    max_tokens = 10*1024*1024  # 10M
    for lang in languages:
        data_dict_all[lang] = []
        textbook_dir = f'{data_root}/MMedC/MMedC/medical_textbooks/{lang.lower()}_ebooks_txt'
        paths = sorted(glob(f'{textbook_dir}/**/*.txt', recursive=True))
        paths = [path for path in paths if len(load_auto(path).split()) > 100000]
        random.shuffle(paths)
        tokens, words = 0, 0
        for path in tqdm(paths, desc=f'Collecting {lang}'):
            ## fake name (path)
            bookname = path.split('/medical_textbooks/')[1].split('.')[0].replace('/', '_')

            content = load_auto(path)
            if lang == 'French': ## TODO: filter / split a whole passage into several paragraphs
                sents = model.split_to_sentence(content)
                if sum([len(sent.split()) < 10 for sent in sents]) / len(sents) > 0.2:
                    continue
            #     content = '\n'.join(' '.join(sents[i:i+5]) for i in range(len(sents)//5))
        
            data_dict_all[lang].append({
                'bookname': bookname,
                'bookname_fake': True,
                'content': content,
            })
            
            # tokens += sum(model.count_token(para) + 1 for para in content.splitlines() if not is_empty(para))
            tokens += model.count_token(content)
            if tokens > max_tokens:
                break

    for lang, books in data_dict_all.items():
        for book_item in tqdm(books, desc=f'Chunking {lang}'):
            book_item['chunks_all'] = model.chunk_content(book_item['content'], 
                                                          max_length=MAX_CORPUS_LEN,
                                                          prefix=book_item['bookname'],
                                                          force_chunk=True)
            del book_item['content']
    
    write_auto(corpus_chunk_path, data_dict_all)


def mmedbench_summ_chunks(corpus_chunk_path):
    data_dict_all: Dict[str, List[Dict[str, Union[str, List[Dict[str, str]]]]]] = \
        load_auto(corpus_chunk_path)
    
    summary_template = (
        "Summarize the following content with NO MORE THAN EIGHT words:\n"
        "```\n{content}\n```\n\n"
        "Now summarize the above content with NO MORE THAN EIGHT words. "
        "Your answer should use the same language (English, 中文, français, 日本語, русск, or español) as the content, and follow this format:\n"
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

    data_to_sum = []
    for lang, books in data_dict_all.items():
        for bi, book in enumerate(tqdm(books, desc=f'Preparing {lang}')):
            for ci, chunk in enumerate(book['chunks_all']):
                content = chunk['data']
                tokens = model.count_token(content)
                max_length = 3600
                if tokens > max_length:
                    tokenized_content = model.tokenizer(content, truncation=False, 
                                                        return_tensors="pt").input_ids[0]
                    half = max_length // 2
                    content = model.tokenizer.decode(tokenized_content[:half], skip_special_tokens=True) + \
                              model.tokenizer.decode(tokenized_content[-half:], skip_special_tokens=True)

                data_to_sum.append([lang, bi, ci, 
                                    summary_template.format(content=content)])
    
    inputs = [x[-1] for x in data_to_sum]
    # TODO: offline - Llama3-70B
    # write_auto('data/tune/MMedBench/mmed_chun_summ_input.jsonl', 
    #            [{'id': i, 'question': q} for i, q in enumerate(inputs)], disp=True)
    
    generator = PreTrainedLLM(GENERATOR_LLM)
    outputs = generator(inputs, batched=True, bs=BATCH_SIZE, 
                        progress_desc='Summarizing Chunks')
    outputs = [_parse_scope(x) for x in outputs]
    for (lang, bi, ci, _), title in zip(data_to_sum, outputs):
        data_dict_all[lang][bi]['chunks_all'][ci]['title'] = title
    
    write_auto(corpus_chunk_path, data_dict_all)


def mmedbench_struct_corpus(corpus_chunk_path, corpus_struct_path):
    data_dict_all: Dict[str, List[Dict[str, Union[str, List[Dict[str, str]]]]]] = \
        load_auto(corpus_chunk_path)

    book2structs: Dict[str, list] = {}    
    for lang, books in data_dict_all.items():
        book2structs[lang] = []
        for book in tqdm(books, desc=f'Preparing {lang}'):
            subsections = []
            for chunk in book['chunks_all']:
                title = chunk['title']
                if is_empty(title):
                    title = chunk['data'].strip().split()[0]
                assert not is_empty(title) and '\n' not in title
                subsections.append(title)
            total_chunk_num = len(book['chunks_all'])
            
            section_content = model.chunk_content('\n'.join(subsections), MAX_STRUCT_LEN)
            section_content = [x['data'].splitlines() for x in section_content]
            assert sum([len(x) for x in section_content]) == total_chunk_num

            section_content = [model.mapping_sentence(x) for x in section_content]
            section_structs = model.batch_forward(section_content, bs=BATCH_SIZE, show_progress=False, 
                                                  return_json=False, input_sent_enum=True)
            section_struct_filtered: List["StructItem"] = []
            for struct_item in section_structs:
                if not struct_item.valid:
                    struct_item = model.generate_struct(struct_item.raw_query)
                struct_json = struct_item.to_json()
                StructItem.prune_struct_level(struct_json, preserve_level=1)
                section_struct_filtered.append(StructItem(struct_dict=struct_json))
            assert section_struct_filtered[0].aspects[0].sent_range[0] == 1
            assert sum(x.aspects[-1].sent_range[1] for x in section_struct_filtered) \
                        == total_chunk_num
            
            section_titles = [section_struct.scope for section_struct in section_struct_filtered]
            sec_names_str = model.mapping_sentence(section_titles)
            book_struct = model.refine_struct_result(sec_names_str, slice_sent_num=40, 
                                                     return_json=True)
            StructItem.prune_struct_level(book_struct, preserve_level=1)
            book_struct = StructItem(struct_dict=book_struct)
            assert book_struct.aspects[0].sent_range[0] == 1
            assert book_struct.aspects[-1].sent_range[1] == len(section_titles)

            if not book.get('bookname_fake', False):
                book_struct.scope = book['bookname']
            
            ## associate chunk contents to book struct
            subsec_off = 0
            for aspect in book_struct.aspects:  # chapter
                assert len(aspect.subaspects) == 0
                st, ed = aspect.sent_range
                for si in range(st-1, ed):  # section
                    section_struct = section_struct_filtered[si]
                    section_aspect, subsec_off = \
                        section_struct.downgrade_to_aspect(True, subsec_off)
                    for subaspect in section_aspect.subaspects:  # subsection
                        sst, sed = subaspect.sent_range
                        assert sed <= subsec_off
                        assert not subaspect.chunks or len(subaspect.chunks) == 0
                        subaspect.chunks = book['chunks_all'][sst-1:sed]
                    aspect.subaspects.append(section_aspect)
            assert subsec_off == total_chunk_num

            book2structs[lang].append(book_struct.to_json())
    
    write_auto(corpus_struct_path, book2structs)


def mmedbench_gen_struct_train(data_root, corpus_struct_path, train_data_path, 
                               mode: Literal['cpt', 'sft']):
    book2structs = load_auto(corpus_struct_path)
    tmp_dict: Dict[str, List["StructItem"]] = {}
    for lang in book2structs.keys():
        book_structs = [StructItem(struct_dict=book_dict) \
                            for book_dict in book2structs[lang]]
        tmp_dict[lang] = book_structs
    book2structs = tmp_dict
    
    vanilla_train_data_all: List[Dict[str, str]] = []
    struct_train_data_all: List[Dict[str, str]] = []

    if mode == 'cpt':
        for lang, books in book2structs.items():
            for book_struct in books:
                bookname = book_struct.scope

                book_dict_temp = book_struct.to_json(disp=True)
                StructItem.prune_struct_level(book_dict_temp, preserve_level=1)
                book_struct_chapter_only = StructItem(struct_dict=book_dict_temp)

                book_struct_cpy = deepcopy(book_struct)
                book_struct_cpy.aspects = []

                for ci, chapter in enumerate(book_struct.aspects):
                    struct_train_data_all.append(
                        model.gen_book_mindmap_corpus(book_struct_chapter_only, chapter,
                                                    idx=f'{bookname}_{ci}_mindmap')
                    )

                    chapter_dict_temp = chapter.upgrad_to_struct().to_json(disp=True)
                    StructItem.prune_struct_level(chapter_dict_temp, preserve_level=1)
                    chapter_struct_section_only = StructItem(struct_dict=chapter_dict_temp)

                    for si, section in enumerate(chapter.subaspects):
                        chapter_struct_cpy = deepcopy(chapter_struct_section_only)
                        chapter_struct_cpy.aspects[si] = section
                        struct_train_data_all.append(
                            model.gen_mindmap_corpus(chapter_struct_cpy, 
                                                    idx=f'{bookname}_{ci}_{si}_mindmap')
                        )

                        for ssi, subsection in enumerate(section.subaspects):
                            assert len(subsection.subaspects) == 0

                            vanilla_train_data_all.append({
                                'text': '\n'.join([chunk['data'] for chunk in subsection.chunks])
                            })

                            for cci, chunk in enumerate(subsection.chunks):
                                struct_train_data_all.append(
                                    model.gen_struct_corpus(chapter_struct_cpy, 
                                                            subsection.upgrad_to_struct(get_raw_query=False), 
                                                            chunk['title'] + '\n' + chunk['data'], 
                                                            idx=f'{bookname}_{ci}_{si}_{ssi}_{cci}')
                                )

    elif mode == 'sft':
        qa_ref_path = f'{data_root}/mmedbench_qa_ref.json'
        if not osp.exists(qa_ref_path):
            index_path = f'{data_root}/mmed_corpus_index'
            qa_ref_all = mmedbench_mapping_qa_to_corpus(book2structs, index_path, data_root)
            write_auto(qa_ref_path, qa_ref_all)
        else:
            qa_ref_all = load_auto(qa_ref_path)

        generator = PreTrainedLLM(GENERATOR_LLM)
        
        prompt_template_ssft_gen = load_auto('src/templates/qa_gen/mmedbench_ssft_gen.md')
        ssft_gen_data_all: List[Dict[str, str]] = []
        prompt_template_ssft2_gen = load_auto('src/templates/qa_gen/mmedbench_ssft+_gen.md')
        ssft2_gen_data_all: List[Dict[str, str]] = []

        for lang, qa_list in qa_ref_all.items():
            ## Vanilla SFT + Struct SFT
            for qi, raw_qa in enumerate(tqdm(qa_list, desc=lang)):
                references: List[Dict[str, str]] = raw_qa['references']

                ref_lang, ref_bi = references[0]['idx'].split('_')[:2]
                book_struct = book2structs[ref_lang][int(ref_bi)]

                selected_chunks = []
                for ref in references:
                    ci, si, ssi, _ = list(map(int, ref['idx'].split('_')[2:]))
                    selected_chunks.append([ci, si, ssi, ref['text']])

                book_struct_cpy, chunk_titles, chunk_contents = \
                    mmedbench_merge_mindmap_path(book_struct, selected_chunks)

                question, answer, rationale = format_demo_qa(raw_qa)
                ssft_gen_data_all.append(
                    model.gen_struct_qa(
                        prompt_template_ssft_gen, book_struct_cpy, 
                        chunk_titles, chunk_contents,
                        idx=f'ssft_{lang}_{qi}', prompt_only=True,
                        question=question, answer=answer, rationale=rationale
                    )
                )
                ssft_gen_data_all[-1]['raw_qa'] = deepcopy(raw_qa)

                ## Vanilla SFT
                prompt, _, _, response = format_trainval_qa(lang, raw_qa, is_with_rationale=False)
                vanilla_train_data_all.append({'prompt': prompt, 'response': response})
                prompt, _, _, response = format_trainval_qa(lang, raw_qa, is_with_rationale=True)
                vanilla_train_data_all.append({'prompt': prompt, 'response': response})

            qa_prompts = [x['question'] for x in ssft_gen_data_all]
            qa_responses = generator(qa_prompts, batched=True, bs=BATCH_SIZE, 
                                     progress_desc='Generating SSFT Samples')
            assert len(ssft_gen_data_all) == len(qa_responses) == len(qa_prompts)
            for raw_data, response in zip(qa_prompts, qa_responses):
                try:
                    assert 'Explanation' in response, response
                    judgement, *_, explanation = response.split('Explanation')
                    struct_flag = 'no' not in judgement.lower()
                except:
                    struct_flag = False

                raw_qa = raw_data['raw_qa']

                if struct_flag:
                    for flag in ['**Integrating Mindmap Elements:**', '**Logical Deduction:**', '**Conclusion:**']:
                        explanation = explanation.replace(flag, '')
                    explanation = explanation[1:] if explanation[0] in PUNCTUATION else explanation
                    explanation = explanation.strip().replace('given', 'related')

                    mindmap = mindmap.replace('**', '')
                    scope = mindmap.split('\n')[0]
                    mindmap = f'Here is the related knowledge points regarding {scope}:\n{mindmap}\n\n'
                    explanation = mindmap + explanation

                    raw_qa['rationale'] = explanation

                prompt, _, _, response = format_trainval_qa(lang, raw_qa, is_with_rationale=False)
                struct_train_data_all.append({'prompt': prompt, 'response': response})
                prompt, _, _, response = format_trainval_qa(lang, raw_qa, is_with_rationale=True)
                struct_train_data_all.append({'prompt': prompt, 'response': response})

            ## Struct SFT extra
            extra_qa_num = int(len(qa_list) * 0.5)
            chapter_num = len(book_struct.aspects)
            qnum_per_ch = extra_qa_num//chapter_num

            path_pool: List[List[List[int]]] = []
            for i in range(chapter_num):
                for _ in range(qnum_per_ch//2):
                    path_num = random.randint(1, 3)
                    cur_paths = []
                    for _ in range(path_num):
                        rand_path = book_struct.random_walk(start=i)
                        if rand_path not in cur_paths:
                            cur_paths.append(rand_path)
                    cur_paths.sort(key=lambda x: '-'.join(list(map(str, x))))
                    if cur_paths not in path_pool:
                        path_pool.append(cur_paths)
            for _ in range(chapter_num*qnum_per_ch//2):
                path_num = random.randint(2, 3)
                cur_paths = []
                for _ in range(path_num):
                    rand_path = book_struct.random_walk()
                    if rand_path not in cur_paths:
                        cur_paths.append(rand_path)
                cur_paths.sort(key=lambda x: '-'.join(list(map(str, x))))
                if cur_paths not in path_pool:
                    path_pool.append(cur_paths)

            for qa_paths in path_pool:
                selected_chunks = []
                assert len(qa_paths) >= 1
                for ci, si, ssi in qa_paths:
                    content = random.choice(
                        book_struct.aspects[ci].subaspects[si].subaspects[ssi].chunks
                    )['data']
                    tokenized_content = model.tokenizer(content, truncation=False, 
                                                    return_tensors="pt").input_ids[0]
                    if len(tokenized_content) > MAX_STRUCT_LEN:
                        half = MAX_STRUCT_LEN // 2
                        content = model.tokenizer.decode(tokenized_content[:half], skip_special_tokens=True) + \
                                    model.tokenizer.decode(tokenized_content[-half:], skip_special_tokens=True)
                    selected_chunks.append([ci, si, ssi, content])
                        
                book_struct_cpy, chunk_titles, chunk_contents = \
                    mmedbench_merge_mindmap_path(book_struct, selected_chunks)
                
                raw_qa = random.choice(qa_list)
                question, answer, rationale = format_demo_qa(raw_qa)
                ssft2_gen_data_all.append(
                    model.gen_struct_qa(
                        prompt_template_ssft2_gen, book_struct_cpy, 
                        chunk_titles, chunk_contents,
                        idx=f'ssft_{lang}_{qi}', prompt_only=True,
                        question=question, answer=answer, rationale=rationale
                    )
                )
                ssft2_gen_data_all[-1]['raw_qa'] = deepcopy(raw_qa)

            qa_prompts = [x['question'] for x in ssft2_gen_data_all]
            qa_responses = generator(qa_prompts, batched=True, bs=BATCH_SIZE, 
                                     progress_desc='Generating SSFT2 Samples')
            assert len(ssft2_gen_data_all) == len(qa_responses) == len(qa_prompts)

            for raw_data, response in zip(qa_prompts, qa_responses):
                try:
                    question, options, answer, explanation = response.split('\n```\n')[1::2]
                except:
                    continue

                for flag in ['**Integrating Mindmap Elements:**', '**Logical Deduction:**', '**Conclusion:**']:
                    explanation = explanation.replace(flag, '')
                explanation = explanation[1:] if explanation[0] in PUNCTUATION else explanation
                explanation = explanation.strip().replace('given', 'related')

                mindmap = mindmap.replace('**', '')
                scope = mindmap.split('\n')[0]
                mindmap = f'Here is the related knowledge points regarding {scope}:\n{mindmap}\n\n'
                explanation = mindmap + explanation

                raw_qa = raw_data['raw_qa']
                raw_qa['rationale'] = explanation
                raw_qa['question'] = question

                prompt, _, _, response = format_trainval_qa(lang, raw_qa, is_with_rationale=False,
                                                            options=options, answer_id=answer)
                struct_train_data_all.append({'prompt': prompt, 'response': response})
                prompt, _, _, response = format_trainval_qa(lang, raw_qa, is_with_rationale=True,
                                                            options=options, answer_id=answer)
                struct_train_data_all.append({'prompt': prompt, 'response': response})

    vanilla_train_data_path = train_data_path
    write_auto(vanilla_train_data_path, vanilla_train_data_all)
    struct_train_data_path = train_data_path.replace(mode, 's' + mode)
    write_auto(struct_train_data_path, struct_train_data_all)


def mmedbench_merge_mindmap_path(book_struct: "StructItem", chunks: List[Union[str, int]]):
    ref_dict_hierarchy: Dict[int, Dict[int, Dict[int, str]]] = {}
    for ci, si, ssi, text in chunks:
        if ci not in ref_dict_hierarchy:
            ref_dict_hierarchy[ci] = {}
        if si not in ref_dict_hierarchy[ci]:
            ref_dict_hierarchy[ci][si] = {}
        if ssi not in ref_dict_hierarchy[ci][si]:
            ref_dict_hierarchy[ci][si][ssi] = ''
        ref_dict_hierarchy[ci][si][ssi] += text + '\n'

    book_struct_cpy = deepcopy(book_struct)
    book_struct_cpy.aspects = []

    chunk_titles, chunk_contents = [], []
    for ci, c_ref in ref_dict_hierarchy.items():
        chapter_cpy = deepcopy(book_struct.aspects[ci])
        chapter_cpy.subaspects = []
        for si, s_ref in c_ref.items():
            section_cpy = deepcopy(book_struct.aspects[ci].subaspects[si])
            section_cpy.subaspects = []
            for ssi, content in s_ref.items():
                subsection = book_struct.aspects[ci].subaspects[si].subaspects[ssi]
                assert len(subsection.subaspects) == 0
                subsection_cpy = deepcopy(subsection)
                subsection_cpy.content = content

                chunk_titles.append(subsection.name)
                chunk_contents.append(content)

                section_cpy.subaspects.append(subsection_cpy)
            chapter_cpy.subaspects.append(section_cpy)
        book_struct_cpy.aspects.append(chapter_cpy)

    return book_struct_cpy, chunk_titles, chunk_contents


def mmedbench_mapping_qa_to_corpus(
    book2structs: Dict[str, List["StructItem"]],
    index_path: str, data_root: str
):
    os.makedirs(index_path, exist_ok=True)
    chunks_all: List[Dict[str, str]] = []
    chunk_size = 1024
    if len(os.listdir(index_path)) == 0:
        for lang, books in book2structs.items():
            for bi, book_struct in enumerate(tqdm(books, desc=f'Preparing {lang}')):
                for ci, chapter in enumerate(book_struct.aspects):
                    for si, section in enumerate(chapter.subaspects):
                        for ssi, subsection in enumerate(section.subaspects):
                            for cci, chunk in enumerate(subsection.chunks):
                                # prefix = f'{lang}_{bi}_{ci}_{si}_{ssi}_{cci}'
                                # chunks_all.extend(model.chunk_content(chunk['data'], max_length=chunk_size,
                                #                                       prefix=prefix, force_chunk=True))
                                chunk['idx'] = f'{lang}_{bi}_{ci}_{si}_{ssi}_{cci}'
                                chunks_all.append(chunk)  # auto chunk
    
    chunkid2text = {x['idx']: x['data'] for x in chunks_all}

    retriever = LlamaIndexRetriever(chunks_all, index_path, chunk_size=chunk_size)

    qa_ref_all: Dict[str, List[Dict[str, str]]] = {}
    for lang in book2structs.keys():
        qa_path = f'{data_root}/MMedBench/MMedBench/Train/{lang}.jsonl'
        qa_list = load_auto(qa_path)
        
        for qa in tqdm(qa_list, desc=lang):
            question = qa['question']

            references = retriever.retrieve(question, return_text=False)
            references: List[Dict[str, str]] = [
                {
                    'idx': ref.node.relationships['1'].node_id,
                    'text': ref.get_text() 
                } for ref in references
            ]

            for ref in references:
                assert ref['text'] in chunkid2text[ref['idx']]
            
            book_indices = ['_'.join(ref['idx'].split('_')[:2]) for ref in references]

            book_counter = Counter(book_indices)
            valid_book, cnt = book_counter.most_common(1)[0]
            if cnt == 1:  # TODO: heuristic
                valid_book = book_indices[-1]
            references = [references[i] for i in range(len(references)) \
                                if book_indices[i] == valid_book]
            
            sorted_indices = np.argsort([ref['idx'] for ref in references])
            references = [references[i] for i in sorted_indices]
            
            qa['references'] = references

        qa_ref_all[lang] = qa_list
    
    return qa_ref_all


if __name__ == '__main__':
    preprocess_mmedbench()