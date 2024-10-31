# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
from typing import Any, Dict, List, Mapping, Optional, Literal, Tuple, Union
from copy import deepcopy

from src.utils.str import (
    white_space_fix, is_empty, upper_start, remove_punc, set_punc, f1_score_auto
)
from src.utils.util import check_and_exit
from src.models.struxgpt_base import (
    StruXGPTBase, AspectItemBase, StructItemBase, StructConfig
)


class AspectItem(AspectItemBase):
    def __init__(self, name: Optional[str] = None, sent_range: list = [], 
                       subaspects: List["AspectItem"] = [],
                       aspect_dict: Optional[dict] = None) -> None:
        super().__init__(name=name, sent_range=sent_range, subaspects=subaspects,
                         aspect_dict=aspect_dict)
    
    def load_from_dict(self, aspect_dict: dict):
        self.name = aspect_dict['name']
        self.sent_range = aspect_dict['sent_range']
        self.subaspects = [AspectItem(aspect_dict=subaspect_dict) \
                                for subaspect_dict in aspect_dict['subaspects']]
        self.content = aspect_dict.get('content', None)
        self.chunks = aspect_dict.get('chunks', None)

    def set_descs(self, desc_list: List[str]):
        self.chunks = desc_list

    def get_descs(self, merge=False) -> Optional[str|List[str]]:
        if merge:
            assert self.chunks and len(self.chunks)
            return ' '.join(upper_start(set_punc(x)) for x in self.chunks)
        else:
            return self.chunks
    
    def clear_descs(self):
        self.chunks.clear()
    
    def add_desc(self, desc: str):
        self.chunks.append(desc)
    
    def is_desc_empty(self):
        return len(self.chunks) == 0


class StructItem(StructItemBase):
    def __init__(self, scfg: Optional[StructConfig] = None,
                       raw_query: str = '', raw_response: str = '',
                       struct_dict: Optional[dict] = None):
        self.aspects: List[AspectItem]
        super().__init__(scfg=scfg, raw_query=raw_query, raw_response=raw_response, 
                         struct_dict=struct_dict)
    
    def __str__(self) -> str:
        if not self.valid:
            return f'Invalid item: {self.dict}'
        
        ret = []
        ret.append(f'## Statement\'s scope:')
        ret.append(f'```{self.scope}```')
        ret.append('')
        ret.append(f'## Statement\'s main aspects and corresponding descriptions:')
        ret.append('```')
        for ai, aspect in enumerate(self.aspects):
            ret.append(f'{ai+1}. {aspect.name}')
            for di, desc in enumerate(aspect.get_descs()):
                ret.append(f'    {ai+1}.{di+1}. {desc}')
        ret.append('```')
        ret.append('')
        return '\n'.join(ret)

    def load_from_dict(self, struct_dict: dict):
        self.scope = struct_dict['scope']
        self.aspects = [AspectItem(aspect_dict=aspect_dict) for aspect_dict in struct_dict['aspects']]
        self.raw_query = struct_dict.get('raw_query', None)
        self.raw_response = struct_dict.get('raw_response', None)
        self.valid = self.scope is not None and len(self.aspects) > 0

    def parse_struct_res(self, **kwargs) -> Tuple[bool, str]:
        if is_empty(self.raw_query):
            return False, 'input_error'
           
        error_type = None
        try:  # check format
            error_type = 'result_spliter'
            scope, hierarchy = self.raw_response.split('```')[-4::2]
            self.scope = white_space_fix(scope)
            assert self.scope, 'result_scope_str'
        except:
            return False, error_type

        def _parse_line(line: str):
            # hierarchy_idx = len(line.split('    ')) - 1
            hierarchy_idx = [c != ' ' for c in line].index(True) // 4
            assert hierarchy_idx < 2, 'result_level_overflow'
            noindent_line = white_space_fix(line)
            indicator, *content = noindent_line.split()
            content = ' '.join(content)
            assert indicator.count('.') == hierarchy_idx + 1, 'result_hierarchy_format'
            assert indicator.endswith('.') and all(int(flag) > 0 for flag in indicator.split('.')[:-1]), 'result_indicator_format'
            return hierarchy_idx, indicator, content
        
        def _add_aspect(name: str, descs: List[str]):
            aspect_item = AspectItem(name=name)
            aspect_item.set_descs(deepcopy(descs))
            self.aspects.append(aspect_item)
        
        try:  # check content
            error_type = 'result_aspect'
            res_lines = hierarchy.strip().splitlines()

            prev_aspect_idx = -1
            prev_aspect_name, prev_aspect_descs = '', []
            for li, line in enumerate(res_lines):
                hierarchy_idx, indicator, content = _parse_line(line)
                indicators = indicator.split('.')[:-1]
                if hierarchy_idx == 0:
                    aspect_idx = int(indicators[0])
                    assert aspect_idx != prev_aspect_idx and aspect_idx >= 1
                    if prev_aspect_idx != -1:
                        assert prev_aspect_name and len(prev_aspect_descs), 'parse_error'
                        _add_aspect(prev_aspect_name, prev_aspect_descs)
                    prev_aspect_idx, prev_aspect_name, prev_aspect_descs = \
                        aspect_idx, content, []
                else:
                    aspect_idx, desc_idx = list(map(int, indicators))
                    assert aspect_idx == prev_aspect_idx
                    assert desc_idx == len(prev_aspect_descs) + 1
                    prev_aspect_descs.append(content)

            assert prev_aspect_name and len(prev_aspect_descs)
            _add_aspect(prev_aspect_name, prev_aspect_descs)

        except AssertionError as e:
            return False, str(e)

        except:
            return False, 'Unknown Error'
        
        return True, self.to_json()


    @staticmethod
    def merge_struct(title: Optional[str], struct_list: Union[List["StructItem"], List[dict]]):
        assert len(struct_list)
        if isinstance(struct_list[0], dict):
            struct_list = [StructItem(struct_dict=data_res) for data_res in struct_list]
        if not title:
            title = ', '.join([struct_item.scope for struct_item in struct_list])

        data_dict = {
            'scope': title,
            'aspects': []
        }
        for struct_res in struct_list:
            data_dict['aspects'].extend([aspect.to_json() for aspect in struct_res.aspects])

        return StructItem(struct_dict=data_dict)


class StruXGPT(StruXGPTBase):
    version = 'StruXGPT-v1'

    def __init__(self, cfg_path: str = './config/struxgpt_v1.yaml', **kwargs):
        super().__init__(cfg_path, **kwargs)
    
    def __call__(self, text: str, return_json=True, **kwargs):
        prompt = self.prepare_prompt(text)

        maxOutputLength = kwargs.get('maxOutputLength', self.max_output_length)
        response = self.call_model(prompt, maxOutputLength=maxOutputLength)

        struct_res = StructItem(self.scfg, text, response)

        if return_json:
            return struct_res.valid, struct_res.dict
        else:
            return struct_res
    
    def batch_forward(self, text_list: List[str], return_json=True, **kwargs):
        prompt_list = [self.prepare_prompt(text) for text in text_list]

        maxOutputLength = kwargs.pop('maxOutputLength', self.max_output_length)
        response_list = self.call_model(prompt_list, batched=True, maxOutputLength=maxOutputLength,
                                        **kwargs)

        struct_res_list = [StructItem(self.scfg, text, response) \
                                for text, response in zip(text_list, response_list)]
        
        if return_json:
            return zip(*[[struct_res.valid, struct_res.dict] for struct_res in struct_res_list])
        else:
            return struct_res_list

    def prepare_prompt(self, text: str, **kwargs):
        return self.map_prompt_template(statement=text)
        
    @staticmethod
    def check_struct_item(item, classtype=StructItem):
        if isinstance(item, dict):
            if classtype is StructItem:
                item = StructItem(struct_dict=item)
            elif classtype is AspectItem:
                item = AspectItem(aspect_dict=item)
            else:
                raise NotImplementedError(str(classtype))
        else:
            assert isinstance(item, classtype), type(item)
        
        return item

    def check_struct_missing_desc(self, struct_item: StructItem, ignore_single_para=True, min_words_per_para=10):
        assert struct_item.valid and struct_item.raw_query and len(struct_item.aspects)
        raw_paras = [para for para in struct_item.raw_query.splitlines() \
                     if len(white_space_fix(para).split()) > min_words_per_para]
        is_missing = False
        if len(raw_paras) == 1 and ignore_single_para:
            return is_missing
        curr_aspect_idx, curr_desc_ind = 0, 0
        for pi, para in enumerate(raw_paras):
            for sent in self.split_to_sentence(para):
                total_recalls = []
                for ai, aspect in enumerate(struct_item.aspects):
                    recalls = [f1_score_auto(prediction=desc, ground_truth=sent, term='recall') \
                                    for desc in aspect.get_descs()]
                    total_recalls.append(recalls if len(recalls) else [0])
                max_recall_per_aspect = np.array([max(recalls) for recalls in total_recalls])

                ## choose target aspect between current and next aspect
                max_recall_per_aspect[:curr_aspect_idx] = 0.
                max_recall_per_aspect[curr_aspect_idx+2:] = 0.

                if max(max_recall_per_aspect) > 0.5:
                    max_aspect_idx = np.argmax(max_recall_per_aspect)
                    curr_aspect_idx = max(curr_aspect_idx, min(max_aspect_idx, curr_aspect_idx+1))
                    curr_aspect_idx = min(curr_aspect_idx, len(struct_item.aspects)-1)
                    curr_desc_ind = np.argmax(total_recalls[curr_aspect_idx])
                else:
                    is_missing = True
                    descs = struct_item.aspects[curr_aspect_idx].get_descs()
                    descs = descs[:curr_desc_ind] + [sent] + descs[curr_desc_ind:]
                    struct_item.aspects[curr_aspect_idx].set_descs(descs)
    
        return is_missing

    def remapping_struct_source(self, struct_item: StructItem):
        assert struct_item.valid and struct_item.raw_query and len(struct_item.aspects)
        raw_paras = [para for para in struct_item.raw_query.splitlines() \
                     if len(white_space_fix(para))]
        
        new_aspect_descs: List[List[str]] = [[] for _ in range(len(struct_item.aspects))]
        curr_aspect_idx = 0
        for pi, para in enumerate(raw_paras):
            total_recalls = []
            for ai, aspect in enumerate(struct_item.aspects):
                # recalls = [f1_score_auto(prediction=desc, ground_truth=para, term='recall') \
                #                 for desc in aspect.get_descs()]
                # total_recalls.append(recalls if len(recalls) else [0])
                total_recalls.append([f1_score_auto(prediction=aspect.get_descs(merge=True), 
                                                    ground_truth=para, term='recall')])
            max_recall_per_aspect = np.array([max(recalls) for recalls in total_recalls])

            ## choose target aspect between current and next aspect
            max_recall_per_aspect[:curr_aspect_idx] = 0.
            max_recall_per_aspect[curr_aspect_idx+2:] = 0.

            max_aspect_idx = np.argmax(max_recall_per_aspect)

            new_aspect_descs[max_aspect_idx].extend(self.split_to_sentence(para))

            curr_aspect_idx = max_aspect_idx
        
        for ai, descs in enumerate(new_aspect_descs):
            struct_item.aspects[ai].set_descs(descs)
        
        struct_item.aspects = [aspect for aspect in struct_item.aspects if not aspect.is_desc_empty()]


    def convert_struct_output(self, struct_item: Union[StructItem, str],
                                    hi:int = 0, fi:int = 1, max_length=None) -> List[str]:
        struct_item = self.check_struct_item(struct_item, StructItem)
        if max_length is None:
            return [struct_item.convert_struct_output(hi=hi, fi=fi)]
        
        def _get_aspect_len(ai) -> int:
            return self.count_token(struct_item.aspects[ai].get_descs(merge=True))

        ## Dual Pointer
        res = []
        a1 = 0
        aspect_num = len(struct_item.aspects)
        assert aspect_num > 0
        while a1 < aspect_num:  # exit if a1 reaches the end aspect
            curr_len = _get_aspect_len(a1)

            ## initialize a2 in each step, incase a1 = aspect_num - 1
            a2 = a1 + 1  
            for a2 in range(a1 + 1, aspect_num):
                ## a2 is unreachable
                next_len = _get_aspect_len(a2)
                curr_len += next_len
                if curr_len > max_length:
                    break

            struct_tmp = deepcopy(struct_item)
            struct_tmp.aspects = struct_tmp.aspects[a1:a2]
            res.append(struct_tmp.convert_struct_output(hi=hi, fi=fi))

            a1 = a2
        
        return res
    