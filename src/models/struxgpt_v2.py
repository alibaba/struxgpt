# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import re
from typing import Any, Dict, List, Mapping, Optional, Literal, Tuple, Union
from copy import deepcopy
import random
import yaml
from multiprocessing import Pool

from src.utils.io import load_file, load_jsonl
from src.utils.str import white_space_fix, is_zh, is_empty, upper_start, remove_punc, set_punc
from src.utils.util import multiprocess_call
from src.models.struxgpt_base import StruXGPTBase, AspectItemBase, StructItemBase, StructConfig

STRUCT_CORPUS_TMPL_POOL = [
    (
        "In the realm of `{field}`, a conceptual mindmap is depicted using a tree-like structure "
        "to represent hierarchical relationships and thematic branches:\n\n"
        "```\n{mindmap}\n```\n\n"
        "Within this organized layout of `{field}`, the detailed subsection on `{aspect}` is described as:\n\n"
    ),
    (
        "The area of `{field}` unfolds into a rich and detailed structure, encapsulating a diverse array of topics and their interconnections. "
        "These topics are organized in a manner that reflects their relationships and thematic relevance to one another, depicted through a structured diagram:\n\n"
        "```\n{mindmap}\n```\n\n"
        "Within this elaborate organization, the concept of `{aspect}` serves as a detailed exploration into a specific element of `{field}`:\n\n"
    ),
    (
        "The `{field}` sector is structured through a complex network of concepts and categories, "
        "as reflected in the following outlined representation:\n\n"
        "```\n{mindmap}\n```\n\n"
        "Zooming in on a discrete element of this intellectual landscape, the topic tagged as `{aspect}` "
        "covers specific subject matter related to `{field}`:\n\n"
    ),
    (
        "Exploring the `{field}`, structured insights reveal a network of thematic areas. "
        "The essence is captured in a concise diagram:\n\n"
        "```\n{mindmap}\n```\n\n"
        "A closer look at the portion labeled `{aspect}` unveils a segment rich in detail, contributing "
        "to the broader understanding of `{field}`:\n\n"
    ),
    (
        "`{field}` encompasses a diverse array of themes, organized for clarity. "
        "The visual schema below illustrates this organization:\n\n"
        "```\n{mindmap}\n```\n\n"
        "Investigating `{aspect}` furnishes insight into a selected theme within `{field}`, enriching the overall comprehension:\n\n"
    ),
    (
        "Contextualizing within the broader spectrum of `{field}`, the organizational structure is delineated as follows:\n\n"
        "```\n{mindmap}\n```\n\n"
        "Delving into `{aspect}`, an integral component of the `{field}` fabric, enriches the grasp of the thematic variety and depth.\n\n"
    ),
    (
        "Within the expansive knowledge area of `{field}`, an organizational schema is represented as:\n\n"
        "```\n{mindmap}\n```\n\n"
        "Exploring `{aspect}` reveals a critical facet of `{field}`, offering insights into its thematic diversity and detail.\n\n"
    ),
    (
        "The discipline of `{field}` is encapsulated by a series of interlinked concepts, mapped out as:\n\n"
        "```\n{mindmap}\n```\n\n"
        "The segment labeled `{aspect}` delves into a particular topic within `{field}`, "
        "illuminating a slice of the broader intellectual landscape:\n\n"
    ),
    (
        "Navigating through `{field}`, one encounters a structured depiction of knowledge as illustrated below:\n\n"
        "```\n{mindmap}\n```\n\n"
        "Within this schema, `{aspect}` serves as a gateway to a distinct area of interest, "
        "shedding light on specific aspects of `{field}`:\n\n"
    ),
    (
        "Diving into the `{field}` landscape, a coherent outline presents itself, showcasing the interconnectedness of its themes:\n\n"
        "```\n{mindmap}\n```\n\n"
        "Focusing on the aspect of `{aspect}`, it serves as a focal point into nuanced exploration within the vast `{field}` territory:\n\n"
    ),
    (
        "The sphere of `{field}` unfolds as a network of insights and principles, outlined for comprehensive understanding:\n\n"
        "```\n{mindmap}\n```\n\n"
        "The exploration of `{aspect}` unveils a segment pivotal to the fabric of `{field}`, providing a perceiving lens:\n\n"
    ),
    (
        "As we chart the terrain of `{field}`, a constellation of concepts emerges, graphically represented as follows:\n\n"
        "```\n{mindmap}\n```\n\n"
        "Focusing on the component marked as `{aspect}`, we uncover layers within `{field}` that resonate with both breadth and depth, offering a panoramic view into the diverse thought processes and methodologies encapsulated within.\n\n"
    ),
    (
        "`{field}` is organized into various key areas, as shown in the diagram below:\n\n"
        "```\n{mindmap}\n```\n\n"
        "`{aspect}` highlights a core area, integral for understanding the overall structure of `{field}`:\n\n"
    ),
    (
        "The structure of `{field}` is detailed below:\n\n"
        "```\n{mindmap}\n```\n\n"
        "A deeper understanding of `{field}` can be achieved by examining `{aspect}`, a vital element of its framework:\n\n"
    ),
    (
        "Overview of `{field}`'s foundational structure is as follows:\n\n"
        "```\n{mindmap}\n```\n\n"
        "Exploring `{aspect}` reveals its crucial role in comprehending the comprehensive schema of `{field}`:\n\n"
    ),
    (
        "`{field}` encompasses a range of interconnected topics, illustrated in the diagram below:\n\n"
        "```\n{mindmap}\n```\n\n"
        "The examination of `{aspect}` provides insight into how key concepts within `{field}` are interrelated:\n\n"
    ),
    (
        "Key elements within `{field}` can be organized as follows:\n\n"
        "```\n{mindmap}\n```\n\n"
        "Investigating the component of `{aspect}` is essential for grasping the complex dynamics in the `{field}` realm:\n\n"
    ),
    (
        "The `{field}` includes various components as detailed in the following structure:\n\n"
        "```\n{mindmap}\n```\n\n"
        "Focusing on `{aspect}` offers an opportunity to explore one of the numerous elements that comprise the `{field}`:\n\n"
    ),
    (
        "Within the scope of `{field}`, multiple dimensions unfold as depicted below:\n\n"
        "```\n{mindmap}\n```\n\n"
        "Delving into `{aspect}` contributes to a broader understanding of the diverse elements that construct the landscape of `{field}`:\n\n"
    ),
    (
        "Comprehensive knowledge of `{field}` can be achieved by examining its individual components, as depicted below:\n\n"
        "```\n{mindmap}\n```\n\n"
        "An exploration of `{aspect}` sheds light on its unique contribution to the `{field}`:\n\n"
    )
]


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

    def set_content(self, content: str) -> None:
        self.content = content

    def get_content(self) -> str:
        return self.content

    def upgrad_to_struct(self, prefix='', get_raw_query=True) -> "StructItem":
        struct_dict = {
            'scope': prefix + self.name, 
            'aspects': [
                AspectItem.offset_sent_range(subaspect.to_json(), 1-self.sent_range[0]) \
                    for subaspect in self.subaspects
                ],
            'raw_query': ' '.join(self.chunks) if get_raw_query and self.chunks else None
        }
        
        return StructItem(struct_dict=struct_dict)

    @staticmethod
    def offset_sent_range(aspect_dict, offset):
        s1, s2 = aspect_dict['sent_range']
        aspect_dict['sent_range'] = [s1 + offset, s2 + offset]
        aspect_dict['subaspects'] = [
            AspectItem.offset_sent_range(subaspect, offset=offset) for subaspect in aspect_dict['subaspects']
        ]
        return aspect_dict
    
    def extend_aspect_hierarchy(self, aspect_dict_list: List["AspectItem"]):
        assert len(self.subaspects) == 0
        offset = self.sent_range[0] - 1
        self.subaspects = [
            AspectItem(aspect_dict=AspectItem.offset_sent_range(subaspect.to_json(), offset=offset)) \
                for subaspect in aspect_dict_list
        ]

    def set_descs(self, sentence_list: List[str]):
        start, end = self.sent_range
        self.chunks = deepcopy(sentence_list[start-1:end])
        for subaspect in self.subaspects:
            subaspect.set_descs(sentence_list)

    def get_descs(self, merge=False) -> Union[str, List[str]]:
        if merge:
            assert self.chunks and len(self.chunks)
            return ' '.join(upper_start(set_punc(x)) for x in self.chunks)
        else:
            return self.chunks

class StructItem(StructItemBase):
    def __init__(self, scfg: Optional[StructConfig] = None,
                       raw_query: str = '', raw_response: str = '', single_sent_ok: bool = True,
                       struct_dict: Optional[dict] = None):
        self.aspects: List[AspectItem]
        super().__init__(scfg=scfg, raw_query=raw_query, raw_response=raw_response, 
                         struct_dict=struct_dict)

    def __str__(self) -> str:
        if not self.valid:
            return f'Invalid item: {self.dict}'

        if self.scfg and self.scfg.format_type == 'JSON':
            s = json.dumps(self.to_json(disp=True), indent=2, ensure_ascii=False)
            pattern = r"\[([\s\d\,]*)]"
            for x in re.findall(pattern, s):
                s = s.replace(x, ' '.join(x.split()))
            s = f'```json\n{s}```'
        else:  # default
            s = self.to_yaml()
            s = f'```yaml\n{s}```'

        return s

    def load_from_dict(self, struct_dict: dict):
        self.scope = struct_dict['scope']
        self.aspects = [AspectItem(aspect_dict=aspect_dict) for aspect_dict in struct_dict['aspects']]
        self.raw_query = struct_dict.get('raw_query', None)
        self.raw_response = struct_dict.get('raw_response', None)
        self.valid = self.scope is not None and len(self.aspects) > 0

    def _delete_subaspect_with_single_sent(self, aspects: List[AspectItem]) -> List[AspectItem]:
        aspects_res = []
        for aspect in aspects:
            if min(aspect.sent_range) < max(aspect.sent_range):
                if len(aspect.subaspects):
                    aspect.subaspects = self._delete_subaspect_with_single_sent(aspect.subaspects)
                aspects_res.append(aspect)

        return aspects_res

    def parse_aspect(self, aspect: dict):
        aspect_content = aspect[self.scfg.aspect_key]
        assert isinstance(aspect_content, str) and aspect_content

        sent_range = aspect[self.scfg.sent_key]
        if isinstance(sent_range, dict):
            sent_range = [sent_range['start'], sent_range['end']]
        else:
            assert isinstance(sent_range, list) and len(sent_range)
        assert all(isinstance(sent_num, int) for sent_num in sent_range)
        assert sent_range[0] <= sent_range[-1]
        sent_range = [min(sent_range), max(sent_range)]

        subaspects = [self.parse_aspect(subaspect) \
                            for subaspect in aspect.get(self.scfg.subaspect_key, [])]
        
        return AspectItem(aspect_content, sent_range, subaspects)

    def parse_struct_res(self, **kwargs):
        single_sent_ok = kwargs.get('single_sent_ok', True)
        # assert all(line.startswith('- Sentence ') for line in raw_query.splitlines()), raw_query
        if not all(line.startswith('- Sentence ') for line in self.raw_query.splitlines()):
            return False, 'input_error'
           
        query_lines = self.raw_query.splitlines()

        error_type = None
        try:  # check format
            error_type = 'result_spliter'
            if self.scfg.format_type == 'JSON':
                response = self.raw_response.split('```json\n')[1].split('\n```')[0]
                error_type = 'format_json'
                response = json.loads(response.replace('\\', '\\\\'))
            elif self.scfg.format_type == 'YAML':
                response = self.raw_response.split('```yaml\n')[1].split('\n```')[0]
                error_type = 'format_yaml'
                response = yaml.safe_load(response)
            else:
                raise ValueError('Uncognized format:', self.scfg.format_type)
            error_type = 'result_scope'
            self.scope = response[self.scfg.scope_key]
            error_type = 'result_scope_str'
            assert isinstance(self.scope, str) and self.scope
            error_type = 'result_aspect'
            self.aspects = [
                self.parse_aspect(aspect) \
                    for aspect in response[self.scfg.struct_key]
            ]
        except:
            self.scope = None
            self.aspects = []
            return False, error_type

        try:  # check content
            ### step1. reset sentence range for the first-level aspects
            for aspect in self.aspects:
                sent_range = aspect.sent_range
                for subaspect in aspect.subaspects:
                    sent_range.extend(subaspect.sent_range)
                aspect.sent_range = [min(sent_range), max(sent_range)]
            total_sent_num = len(query_lines)
            if any(flag in self.aspects[-1].name.lower() for flag in ['external link', 'reference', 'conclusion', 'see also'])\
                    and self.aspects[-1].sent_range[-1] <= total_sent_num:
                self.aspects[-1].sent_range[-1] = total_sent_num
            assert self.aspects[0].sent_range[0] == 1, 'range_start'
            assert self.aspects[-1].sent_range[-1] == total_sent_num, 'range_end'
            sent_prev = 0
            for aspect in self.aspects:
                sent_range = aspect.sent_range
                assert sent_range[0] == sent_prev + 1, 'range_middle'
                sent_prev = sent_range[1]
            ### step2. delete subaspect with single sentence
            for aspect in self.aspects:
                aspect.subaspects = self._delete_subaspect_with_single_sent(aspect.subaspects)
            ### step3. check result with single sentence for all aspects
            if not single_sent_ok:
                single_flag = True
                for aspect in self.aspects:
                    if aspect.sent_range[0] != aspect.sent_range[1]:
                        single_flag = False
                        break
                assert not single_flag, 'single_sent_aspect'

        except AssertionError as e:
            self.scope = None
            self.aspects = []
            return False, str(e)
        
        return True, self.to_json()

    def downgrade_to_aspect(self, keep_subaspects=False, sent_off=0) -> Tuple["AspectItem", int]:
        first_aspect, last_aspect = self.aspects[0], self.aspects[-1]
        first_sent, last_sent = first_aspect.sent_range[0], last_aspect.sent_range[1]
        super_aspect = {
            'name': self.scope, 
            'sent_range': [first_sent+sent_off, last_sent+sent_off],
            'subaspects': [],
            'chunks': StruXGPT.remapping_sentence(self.raw_query)
        }
        if keep_subaspects:
            super_aspect['subaspects'] = [
                AspectItem.offset_sent_range(aspect.to_json(), offset=sent_off) \
                    for aspect in self.aspects
            ]
        super_aspect = AspectItem(aspect_dict=super_aspect)

        return super_aspect, sent_off+last_sent

    @staticmethod
    def lazzy_struct(title: str, sent_list: List[str]):
        sent_num = len(sent_list)
        data_dict = {
            'scope': title,
            'aspects': [
                {
                    'name': title,
                    'sent_range': [1, sent_num],
                    'subaspects': []
                }
            ],
            'raw_query': StruXGPT.mapping_sentence(sent_list),
            # 'raw_response': None
        }
        return StructItem(struct_dict=data_dict)

    @staticmethod
    def merge_struct(title: str, struct_list: Union[List["StructItem"], List[dict]],
                     mode: Literal["concat", 'upgrade']):
        assert len(struct_list)
        if isinstance(struct_list[0], dict):
            struct_list = [StructItem(struct_dict=data_res) for data_res in struct_list]

        data_dict: Dict[str, Union[str, List[str]]] = {
            'scope': title,
            'aspects': [],
            'raw_query': [],
            'raw_response': None
        }
        sent_num_offset = 0
        for struct_res in struct_list:
            data_dict['raw_query'].extend(StruXGPT.remapping_sentence(struct_res.raw_query))

            sent_num = struct_res.aspects[-1].sent_range[1]
            if mode == 'concat':
                for aspect in struct_res.aspects:
                    data_dict['aspects'].append(
                        AspectItem.offset_sent_range(aspect.to_json(), sent_num_offset)
                    )
            elif mode == 'upgrade':
                super_aspect = {
                    'name': struct_res.scope,
                    'sent_range': [
                        sent_num_offset + 1, 
                        sent_num_offset + sent_num
                    ],
                    'subaspects': [
                        AspectItem.offset_sent_range(aspect.to_json(), sent_num_offset) \
                            for aspect in struct_res.aspects
                    ]
                }
                data_dict['aspects'].append(super_aspect)
            else:
                raise NotImplementedError(mode)
            
            sent_num_offset += sent_num
        
        data_dict['raw_query'] = StruXGPT.mapping_sentence(data_dict['raw_query'])

        return StructItem(struct_dict=data_dict)

    @staticmethod
    def prune_struct_level(struct_dict: dict, is_root=True, preserve_level=1):
        if preserve_level == 0 or preserve_level < -1:
            return
        aspect_key = 'aspects' if is_root else 'subaspects'
        if len(struct_dict[aspect_key]) == 0:
            return
        
        for aspect in struct_dict[aspect_key]:
            if preserve_level == 1:
                aspect['subaspects'] = []
            elif preserve_level == -1 and len(aspect['subaspects']) == 1:
                aspect['name'] = aspect['subaspects'][0]['name']
                aspect['subaspects'] = []
            else:
                for subaspect in aspect['subaspects']:
                    StructItem.prune_struct_level(subaspect, False, preserve_level-1)

    def random_walk(self, start: Optional[int] = None) -> List[int]:
        path = []
        assert len(self.aspects)
        ai: int = start or random.choice(list(range(len(self.aspects))))
        aspect = self.aspects[ai]
        path.append(ai)
        while len(aspect.subaspects):
            ai: int = random.choice(list(range(len(aspect.subaspects))))
            aspect = aspect.subaspects[ai]
            path.append(ai)
        return path

    def set_aspect_descs(self):
        assert not is_empty(self.raw_query)
        sentence_list = StruXGPT.remapping_sentence(self.raw_query)
        for aspect in self.aspects:
            aspect.set_descs(sentence_list=sentence_list)


class StruXGPT(StruXGPTBase):
    version = 'StruXGPT-v2'

    def __init__(self, cfg_path: str = './config/struxgpt_v2.yaml', **kwargs):
        super().__init__(cfg_path, **kwargs)
    
    def __call__(self, text: str, return_json=True, **kwargs):
        prompt, sentences_str = self.prepare_prompt(text, return_sentences_str=True)

        maxOutputLength = kwargs.get('maxOutputLength', self.max_output_length)
        response = self.call_model(prompt, maxOutputLength=maxOutputLength)

        struct_res = StructItem(self.scfg, sentences_str, response)

        if return_json:
            return struct_res.valid, struct_res.dict
        else:
            return struct_res
    
    def batch_forward(self, text_list: List[str], return_json=True, **kwargs):
        prompt_list, sentences_str_list = \
            zip(*[self.prepare_prompt(text, return_sentences_str=True, **kwargs) for text in text_list])

        maxOutputLength = kwargs.pop('maxOutputLength', self.max_output_length)
        response_list = self.call_model(prompt_list, batched=True, maxOutputLength=maxOutputLength,
                                        **kwargs)

        struct_res_list = [StructItem(self.scfg, sentences_str, response) \
                                for sentences_str, response in zip(sentences_str_list, response_list)]
        if return_json:
            return zip(*[[struct_res.valid, struct_res.dict] for struct_res in struct_res_list])
        else:
            return struct_res_list

    def prepare_prompt(self, text: str, **kwargs):
        if kwargs.get('input_sent_enum', False):
            sentences_str = text
        else:
            sentence_list = []
            for para in text.splitlines():
                if len(white_space_fix(para)) == 0:
                    continue
                sentence_list.extend(self.split_to_sentence(para))
            sentences_str = self.mapping_sentence(sentence_list)
        prompt = self.map_prompt_template(sentences=sentences_str)
        
        if kwargs.get('return_sentences_str', False):
            return prompt, sentences_str
        else:
            return prompt

    @staticmethod    
    def mapping_sentence(sentence_list: List[str]):
        template = '- Sentence {}. {}'
        formatted_sent_list = [template.format(idx+1, sent) for idx, sent in enumerate(sentence_list)]
        return "\n".join(formatted_sent_list)

    @staticmethod    
    def remapping_sentence(sentences_str: str):
        sentence_list = sentences_str.splitlines()
        sentence_list = ['. '.join(sent.split('. ')[1:]) for sent in sentence_list]
        return sentence_list

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
            assert isinstance(item, classtype)
        
        return item

    def refine_struct_format(self, raw_query: str, raw_response: str, format: Literal["json", "yaml"],
                                   return_json: bool = True, verbose=False):
        refine_prompt = (
            f"Correct any syntax error in the following {format.upper()} snippet and format it properly. "
            f"Ensure that the corrected {format.upper()} is valid and retains the same information as the original. "
            f"The corrected {format.upper()} should be displayed as code, starting with \"```{format}\" and ending with \"```\".\n\n"
            f"Here is the {format.upper()} snippet to correct:\n"
            f"{raw_response}"
        )
        try:
            refined_response = self.call_model(refine_prompt)
            struct_res = StructItem(self.scfg, raw_query, refined_response, single_sent_ok=True)
            assert struct_res.valid
            if verbose:
                print(f'Successfully refined {format.upper()} response.')
            return struct_res.to_json() if return_json else struct_res
        except:
            if verbose:
                print(f'Failed to refine {format.upper()} response. Try to structurize again.')
            # print(refined_response)
            return self.refine_struct_result(raw_query, return_json=return_json)

    def generate_struct(self, raw_query: str, timeout: int = 3, verbose=False) -> "StructItem":
        generate_params = {
            "temperature": 0.2 if timeout > 1 else 0.0,
            "maxWindowSize": 4096,
            "maxOutputLength": 2048,
            "system": self.prompt_system,
        }
        prompt = self.prepare_prompt(raw_query, input_sent_enum=True)

        for _ in range(timeout):
            try:
                response = self.call_model(prompt, **generate_params)
                struct_res = StructItem(self.scfg, raw_query, response, single_sent_ok=True)
                assert struct_res.valid
                if verbose:
                    print('Successfully refined structurization result.')
                return struct_res
            except:
                if verbose:
                    print('Failed to structurize. Try again.')
                # print(f'##Response\n{response}\n\n')
                # exit()
        
        if verbose:
            print('Structurization timeout. Downgrade to summatization.')
        return self.lazzy_struct_by_sum(raw_query, return_json=False)

    def refine_struct_result(self, raw_query: str, slice_sent_num: int = 0, num_threads: int = 0,
                                   timeout: int = 3, return_json: bool = True):
        if slice_sent_num == 0:
            struct_res = self.generate_struct(raw_query, timeout=timeout)
        else:
            assert isinstance(slice_sent_num, int) and slice_sent_num > 0
            sent_list = self.remapping_sentence(raw_query)
            ns, rs = len(sent_list), slice_sent_num
            if ns > rs:
                if num_threads > 1:
                    struct_args_list = []
                    for i in range(0, ns, rs):
                        sent_sublist = sent_list[i:i+rs]
                        sentences_str = self.mapping_sentence(sent_sublist)
                        struct_args_list.append((sentences_str, timeout))
                    struct_res_list = multiprocess_call(self.generate_struct, struct_args_list, 
                                                        num_threads=min(num_threads, len(struct_args_list)))
                else:
                    struct_res_list = []
                    for i in range(0, ns, rs):
                        sent_sublist = sent_list[i:i+rs]
                        sentences_str = self.mapping_sentence(sent_sublist)
                        struct_res_list.append(self.generate_struct(sentences_str, timeout=timeout))
                scope = self.summarize_sentence_scope(raw_query)
                struct_res = StructItem.merge_struct(scope, struct_res_list, mode='concat')
            else:
                struct_res = self.generate_struct(raw_query, timeout=timeout)
        
        return struct_res.to_json() if return_json else struct_res

    def lazzy_struct_by_sum(self, raw_query: str, return_json: bool = True,
                            raw_scope: str = ''):
        scope = self.summarize_sentence_scope(raw_query, raw_scope=raw_scope)
        sent_list = self.remapping_sentence(raw_query)
        struct_res = StructItem.lazzy_struct(scope, sent_list)
        return struct_res.to_json() if return_json else struct_res

    def summarize_sentence_scope(self, raw_query: str, timeout: int = 5, 
                                 raw_scope: str = '', verbose=False):
        input_prompt = (
            "Summarize a union scope (central theme) within ten words to cover all the given sentences:\n"
            f"```\n{raw_query}\n```\n\n"
            "Your answer should use the same language as the senteces, and follow this format:\n"
            "```[scope goes here]```"
        )
        generate_params = {
            "temperature": 0.1,
            "maxOutputLength": 128,
            "system": "You are a helpful assistant."
        }
        # scope_backup = self.remapping_sentence(raw_query)[0]
        
        scope = ''
        for i in range(timeout):
            try:
                if i == 0 and raw_scope:
                    response = raw_scope
                else:
                    if verbose:
                        print('Model called...')
                    response = self.call_model(input_prompt, **generate_params)
                if '[' in response and ']' in response:
                    # scope = white_space_fix(re.findall('\[(.*)\]', response)[0])
                    candidates = [white_space_fix(cand) for cand in re.findall('\[(.*)\]', response)]
                    candidates = [cand for cand in candidates if not is_empty(cand)]
                    assert len(candidates)
                    scope = candidates[0]
                elif '```' in response:
                    scope = white_space_fix(response.split('```')[-2])
                    assert not is_empty(scope)
                elif '"' == response[0] and '"' == response[-1] and response.count('"') == 2:
                    scope = response[1:-1]
                elif raw_scope:
                    # print('Use the additional candidate:', raw_scope)
                    scope = raw_scope
                else:
                    if i < timeout - 1:
                        if verbose:
                            print('Failed to summarize. Try again.', response)
                        raise RuntimeError
                    else:
                        scope = response # if not is_empty(response) else scope_backup
                        if verbose:
                            print('Summarization timeout. Current result: ', response, 'Final result: ', scope)
                break
            except:
                continue

        # if is_empty(scope) or len(scope) == 0 or 'scope' in scope.lower():
        #     scope = scope_backup
        # print('Successful summarization:', scope)
        return scope

    def dump_mindmap(self, obj: "StructItem", prefix='', is_root=True, highlights=[]):
        lines = []
        # 定义各种特殊字符，表示树的枝干连接
        branch = '├─ '
        last_branch = '└─ '
        pipe = '│'
        empty = ' '

        # 处理根节点
        if is_root:
            lines.append(prefix + remove_punc(obj.scope))

        # 获取下一层的aspect列表，如果不存在，则为空列表
        for i, aspect in enumerate(obj.aspects):
            # 选择合适的分支符号
            is_last = (i == len(obj.aspects) - 1)
            if is_last:
                new_prefix = prefix + last_branch
                next_prefix = prefix + empty + ' '
            else:
                new_prefix = prefix + branch
                next_prefix = prefix + pipe + ' '

            # 将当前aspect的内容加到文本中
            aspect_name = deepcopy(remove_punc(aspect.name))
            if aspect_name in highlights:
                aspect_name = f'**{aspect_name}**'
            lines.append(new_prefix + aspect_name)

            # 如果存在子层，递归调用函数处理子层
            if len(aspect.subaspects):
                lines.extend(
                    self.dump_mindmap(aspect.upgrad_to_struct(), next_prefix, 
                                      is_root=False, highlights=highlights)
                )

        # 处理根节点
        if is_root:
            mindmap = '\n'.join(lines)
            for flag in highlights:
                assert f'**{flag}**' in mindmap, f'\n{mindmap}\n\n{flag}'
            return mindmap
        else:
            return lines

    def gen_struct_corpus(self, knowledge_struct: Optional[StructItem | dict], 
                                content_struct: Optional[StructItem | dict], 
                                content_text: str, idx: str):
        knowledge_struct = self.check_struct_item(knowledge_struct, StructItem)
        content_struct = self.check_struct_item(content_struct, StructItem)

        # prompt_template = (
        #     "In the realm of `{field}`, a conceptual mindmap is depicted using a tree-like structure "
        #     "to represent hierarchical relationships and thematic branches:\n\n"
        #     "```\n{mindmap}\n```\n\n"
        #     "Within this organized layout of `{field}`, the detailed subsection on `{aspect}` is described as:\n\n"
        # )
        prompt_template: str = random.choice(STRUCT_CORPUS_TMPL_POOL)
        field = knowledge_struct.scope
        aspect = content_struct.scope
        mindmap = self.dump_mindmap(knowledge_struct, highlights=[aspect,])
        prompt = prompt_template.format(field=field, mindmap=mindmap, aspect=aspect)

        return {'idx': idx, 'prompt': prompt, 'response': content_text}

    def gen_mindmap_corpus(self, knowledge_struct: Optional[StructItem | dict], idx: str):
        knowledge_struct = self.check_struct_item(knowledge_struct, StructItem)

        prompt_template: str = random.choice(STRUCT_CORPUS_TMPL_POOL)
        field = knowledge_struct.scope
        mindmap = self.dump_mindmap(knowledge_struct)
        prompt = prompt_template.split('\n\n')[0].format(field=field)
        
        return {'idx': idx, 'prompt': prompt, 'response': mindmap}

    def gen_book_mindmap_corpus(self, book_struct: Optional[StructItem | dict], 
                                chapter_aspect: Optional[AspectItem | dict], idx: str):
        book_struct = self.check_struct_item(book_struct, StructItem)
        chapter_aspect = self.check_struct_item(chapter_aspect, AspectItem)

        prompt_template: str = random.choice(STRUCT_CORPUS_TMPL_POOL)
        book_mindmap = self.dump_mindmap(book_struct, highlights=[chapter_aspect.name])
        prompt = (
            f"In Medicine, {book_struct.scope} holds significant stature, "
            f"which comprises the following specialized subdomains:\n"
            f"{book_mindmap}\n\n"
        )

        prompt += prompt_template.split('\n\n')[0].format(field=chapter_aspect.name)

        chapter_mindmap = self.dump_mindmap(chapter_aspect.upgrad_to_struct())
        
        return {'idx': idx, 'prompt': prompt, 'response': chapter_mindmap}

    def gen_struct_qa(self, qa_gen_template: str,
                            ctx_struct: Union[StructItem, dict], 
                            chunk_titles: List[str], chunk_contents: List[str], idx: str, 
                            demonstrations: str = "", response: str = "",
                            prompt_only: bool = False, parse_only: bool = False,
                            **prompt_kwargs):
        if not parse_only:
            #### generate qa data
            ctx_struct = self.check_struct_item(ctx_struct, StructItem)
            ctx_mindmap = self.dump_mindmap(ctx_struct, highlights=chunk_titles)
            contents = ''
            for title, content in zip(chunk_titles, chunk_contents):
                contents += f'\n* **{title}**\n```\n{content}\n```\n'
            chunk_num = len(chunk_titles)

            prompt = qa_gen_template.format(mindmap=ctx_mindmap, field=ctx_struct.scope, 
                                            passage_num=chunk_num, contents=contents,
                                            demos=demonstrations, **prompt_kwargs)
            if prompt_only:
                return {'idx': idx, 'question': prompt, 
                        'ctx_struct': ctx_struct, 'ctx_mindmap': ctx_mindmap}
            else:
                response = self.call_model(prompt)
        
        #### parse qa data
        try:
            invalid_keywords = ['test', 'exam', 'mindmap', 'segment',
                                '考', '思维导图', '片段']
            sections = response.strip().split('\n\n')
            assert len(sections) == 3
            assert sections[0].splitlines()[0].startswith('# Question')
            assert sections[1].splitlines()[0].startswith('# Answer')
            assert sections[2].splitlines()[0].startswith('# Explanation')
            question, answer, explanation = [
                '\n'.join(section.splitlines()[1:]) for section in sections
            ]
            explanation = '\n'.join(line for line in explanation.splitlines() \
                                        if all(flag not in line.lower() for flag in invalid_keywords))
        
            return {'idx': idx, 'question': question, 'answer': answer, 'explanation': explanation,
                    'ctx_content': chunk_contents, 'chunk_titles': chunk_titles}
        except:
            return {}

    def convert_struct_output(self, struct_item: Union[StructItem, str],
                                    hi:int = 0, fi:int = 1, max_length=None) -> List[str]:
        struct_item = self.check_struct_item(struct_item, StructItem)
        if not struct_item.aspects[0].get_descs():
            struct_item.set_aspect_descs()
        if max_length is None:
            return [struct_item.convert_struct_output(hi=hi, fi=fi)]
        
        def _get_aspect_len(ai) -> int:
            return self.count_token(struct_item.aspects[ai].get_descs(merge=True))

        ## Dual Pointer && DFS
        res = []
        a1 = 0
        aspect_num = len(struct_item.aspects)
        assert aspect_num > 0
        while a1 < aspect_num:
            curr_len = _get_aspect_len(a1)

            if curr_len > max_length * 1.5 and len(struct_item.aspects[a1].subaspects):
                _struct_item = struct_item.aspects[a1].upgrad_to_struct(prefix=f'{struct_item.scope} ')
                res.extend(self.convert_struct_output(_struct_item, hi=hi, fi=fi, max_length=max_length))
                a1 += 1
                continue

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
    