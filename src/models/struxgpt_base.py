# Copyright (c) Alibaba, Inc. and its affiliates.
import re
import os
import json
import yaml
from tqdm import tqdm
from typing import Dict, List, Optional, Literal, Tuple, Union
import numpy as np
import requests
from http import HTTPStatus

from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import torch

from src.utils import (
    load_file, load_auto,
    is_zh, is_empty, remove_punc, upper_start, PUNCTUATION,
    multiprocess_call
)


class StructConfig(object):
    def __init__(self, format_type="JSON",
                 scope_key="Scope", struct_key="AspectHierarchy",
                 aspect_key="Aspect", sent_key="SentenceRange", 
                 subaspect_key="SubAspects"):
        self.format_type = format_type

        self.scope_key = scope_key
        self.struct_key = struct_key
        self.aspect_key = aspect_key
        self.sent_key = sent_key
        self.subaspect_key = subaspect_key


class AspectItemBase(object):
    def __init__(self, name: Optional[str] = None, sent_range: list = [], 
                       subaspects: List["AspectItemBase"] = [],
                       aspect_dict: Optional[dict] = None) -> None:
        self.content: Optional[str] = None
        self.chunks: Optional[List[str]] = None

        if name is not None:
            assert aspect_dict is None
            self.name = name
            self.sent_range = sent_range
            self.subaspects = subaspects
        else:
            assert aspect_dict is not None
            self.load_from_dict(aspect_dict)
    
    def load_from_dict(self, aspect_dict: dict):
        raise NotImplementedError

    def to_json(self, disp=False):
        res = {
            'name': self.name, 'sent_range': self.sent_range,
            'subaspects': [aspect.to_json(disp=disp) for aspect in self.subaspects]
        }
        if not disp and self.content is not None:
            res['content'] = self.content
        if not disp and self.chunks is not None:
            res['chunks'] = self.chunks
        return res

    def to_yaml(self):
        res =  {'AspectName': self.name, 
                'SentenceRange': {'start': self.sent_range[0], 'end': self.sent_range[1]}}
        if len(self.subaspects):
            res['SubAspects'] = [aspect.to_yaml() for aspect in self.subaspects]

        return res

    def get_descs(self, merge=False) -> Optional[str|List[str]]:
        raise NotImplementedError


class StructItemBase(object):
    def __init__(self, scfg: Optional[StructConfig] = None,
                       raw_query: str = '', raw_response: str = '',
                       struct_dict: Optional[dict] = None,
                       **kwargs):
        self.scfg: Optional[StructConfig] = None
        self.raw_query: str = raw_query
        self.raw_response: str = raw_response

        self.scope: str = None
        self.aspects: List[AspectItemBase] = []

        if scfg is not None:
            assert struct_dict is None
            self.scfg = scfg

            self.valid, self.dict = \
                self.parse_struct_res(**kwargs)
        elif not isinstance(struct_dict, dict):
            self.valid = False
            self.dict = struct_dict
        else:
            self.load_from_dict(struct_dict)
            self.dict = struct_dict
    
    def __str__(self) -> str:
        raise NotImplementedError
    
    def load_from_dict(self, struct_dict: dict):
        raise NotImplementedError

    def to_json(self, disp=False):
        return {
            'scope': self.scope, 
            'aspects': [aspect.to_json(disp=disp) for aspect in self.aspects],
            'raw_query': self.raw_query,
            'raw_response': self.raw_response
        }
    
    def to_yaml(self):
        obj = {'Scope': self.scope, 'Aspects': [aspect.to_yaml() for aspect in self.aspects]}
        return yaml.safe_dump(obj, sort_keys=False, indent=2, default_flow_style=False, allow_unicode=True)

    def is_zh(self):
        def _get_aspect_name(aspect: "AspectItemBase"):
            texts = [aspect.name]
            for subaspect in aspect.subaspects:
                texts.extend(_get_aspect_name(subaspect))
            return texts
            
        texts = [self.scope]
        for aspect in self.aspects:
            texts.extend(_get_aspect_name(aspect))

        return is_zh(' '.join(texts))
    
    def parse_struct_res(self, **kwargs) -> Tuple[bool, str]:
        raise NotImplementedError

    def convert_struct_output(self, hi:int = 0, fi:int = 1) -> str:
        scope = remove_punc(self.scope)
        total_aspects = ', '.join([remove_punc(aspect.name, lowercase=True) for aspect in self.aspects])
        res = ''

        ## head (scope)
        if hi == 0:
            res += f'{upper_start(scope)}.\n'
        elif hi == 1:
            res += f'About {scope}:\n'
        elif hi == 2:
            res += f'{upper_start(scope)} involves {total_aspects}:\n'
        elif hi == 3:
            res += f'{upper_start(scope)} involves several aspects:\n'
        elif hi == 4:
            res += f'{upper_start(scope)} involves several main aspects with corresponding descriptive statements:\n'
        elif hi == 5:
            res += f'This passage talks about {scope}, involving several aspects:\n'
        elif hi == 6:
            res += f'\n'
        else:
            raise NotImplementedError(f'Head {hi}')

        # format (aspects and descriptions)
        if fi == 0:
            # numerical hierarchy
            for ai, aspect in enumerate(self.aspects):
                res += f'{ai+1}. {aspect.name}\n'
                for di, desc in enumerate(aspect.get_descs()):
                    res += f'    {ai+1}.{di+1}. {desc}\n'
            return res
        
        for ai, aspect in enumerate(self.aspects):
            aspect_title = remove_punc(aspect.name, lowercase=True)
            aspect_title = upper_start(aspect_title)
            descs = upper_start(aspect.get_descs(merge=True))

            if fi == 1:
                res += f'{ai+1}. {aspect_title}: {descs}\n'
            elif fi == 2:
                res += f'**{aspect_title}**: {descs}\n'
            else:
                raise NotImplementedError(f'Format {fi}')
        
        res = res[:-1]
        
        return res

class ModelBase(object):
    def __init__(self, model_name_or_path="", model_type="", max_output_length=128,
                       use_vllm=True, prompt_system='You are a helpful assistant.', **kwargs):
        self.model_name_or_path = model_name_or_path
        self.model_type = model_type
        self.use_vllm = use_vllm
        self.prompt_system = prompt_system
        self.max_output_length = max_output_length
        self.model_is_api = self.model_name_or_path.startswith('http')

        if self.model_is_api:
            self.call_model = self.call_model_api  
        else:
            self.call_model = self.call_model_local
            self.build_model_local(**kwargs)

    def __call__(self, prompt: str, **kwargs):
        return self.call_model(prompt, **kwargs)

    def build_model_local(self, debug=False):
        if self.use_vllm:
            if debug:
                self.model = None
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name_or_path,
                    use_fast=False,
                    padding_side="left",
                    trust_remote_code=True,
                )
            else:
                self.model = LLM(
                    self.model_name_or_path,
                    tensor_parallel_size=torch.cuda.device_count(),
                    trust_remote_code=True,
                    # gpu_memory_utilization=0.6  # TODO
                )
                self.tokenizer = self.model.get_tokenizer()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                use_fast=False,
                padding_side="left",
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
    
    def call_model_api(self, prompt: Union[str, list], history=None,
                             batched=False, bs=8, **kwargs) -> Union[str, List[str]]:
        if batched:
            assert isinstance(prompt, list)
            return multiprocess_call(self.call_model_api, 
                                     [(s, ) for s in prompt], 
                                     [kwargs] * len(prompt),
                                     num_threads=bs)
        else:
            assert isinstance(prompt, str)
            model_url = kwargs.get('url', self.model_name_or_path)
            headers = { 'Content-Type': 'application/json' }
            input_param = {
                "input": prompt,
                "history": history,
                # "serviceParams": {
                #     "maxContentRound": 5,
                #     "stream": False,
                #     "generateStyle": "chat"
                # },
                "modelParams": {
                    "best_of": 1,
                    # "temperature": 0.0,
                    "length_penalty": 1.0
                },
                "serviceParams": {
                    # "maxWindowSize": 6144,
                    # "maxOutputLength": 2048,
                    "promptTemplateName": "default",
                    # "system": "You are a helpful assistant."
                },
            }
            input_param['modelParams']['temperature'] = kwargs.get('temperature', 0.0)
            input_param['serviceParams']['maxWindowSize'] = kwargs.get('maxWindowSize', 8192)
            input_param['serviceParams']['maxOutputLength'] = kwargs.get('maxOutputLength', 128)
            input_param['serviceParams']['system'] = kwargs.get('system', "You are a helpful assistant.")
            # print(input_param)
            data = json.dumps(input_param)

            res = requests.request("POST", model_url, headers=headers, data=data)
            if res.status_code != HTTPStatus.OK:
                raise RuntimeError(f"Connect to server error.\nStatus Code: {res.status_code}\nMessage: {res.reason}")
            else:
                response = json.loads(res.text)
                return response['data']['output']

    def call_model_local(self, prompt: Union[str, List[str]], history=None, batched=False, bs=8,
                               template: Literal['default', 'llama2', 'qwen', 'mistral', 'intern2', 'plain']='default',
                               **kwargs) -> Union[str, List[str]]:
        assert history is None, 'Unsupported now.'

        system = kwargs.get('system', self.prompt_system)
        if template == 'default':
            template = self.model_type

        def _format_example(prompt):
            if template == 'llama2':
                prompt = f'[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{prompt} [/INST]'
            elif template == 'mistral':
                prompt = f'[INST] {prompt} [/INST]'
            elif template in ['intern2', 'qwen']:
                prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            elif template == 'plain':
                pass
            else:
                raise NotImplementedError(template)
            
            return prompt
        
        temperature = kwargs.get('temperature', 0.0)
        max_tokens = kwargs.get('maxOutputLength', 128)
        show_progress = kwargs.get('show_progress', True)
        progress_desc = kwargs.get('progress_desc', None)
        stop = kwargs.get('stop', ['</s>', '<|im_end|>'])
        if batched:
            assert self.use_vllm and not isinstance(prompt, str)
            sampling_kwargs = SamplingParams(temperature=temperature, stop=stop, max_tokens=max_tokens)
            pred = []
            pbar = range(0, len(prompt), bs)
            if show_progress:
                pbar = tqdm(pbar, desc=progress_desc)
            for i in pbar:
                questions = [_format_example(qry) for qry in prompt[i:i+bs]]
                outputs = self.model.generate(questions, sampling_kwargs, use_tqdm=False)
                pred.extend([output.outputs[0].text for output in outputs])
        else:
            assert isinstance(prompt, str)
            prompt = _format_example(prompt)
            if self.use_vllm:
                sampling_kwargs = SamplingParams(temperature=temperature, stop=stop, max_tokens=max_tokens)
                outputs = self.model.generate([prompt], sampling_kwargs, use_tqdm=False)
                pred = [output.outputs[0].text for output in outputs][0]
            else:
                inputs = self.tokenizer(prompt, truncation=False, return_tensors="pt").to('cuda:0')
                context_length = inputs.input_ids.shape[-1]
                eos_token_id = [self.tokenizer.eos_token_id]
                for stop_word in stop:
                    eos_token_id.extend(self.tokenizer.encode(stop_word))
                output = self.model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            eos_token_id=eos_token_id,
                        )[0]
                pred = self.tokenizer.decode(output[context_length:], skip_special_tokens=True)

        return pred

    def close(self):
        if self.model_is_api:
            raise Warning('Closing API model is not configured. Please manually comment this line.')
            pass
        else:
            import gc
            import torch
            if self.use_vllm:
                try:
                    from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
                    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
                    destroy_model_parallel()
                except:
                    pass
                del self.model.llm_engine

            del self.model
            # del self
            gc.collect()
            torch.cuda.empty_cache()
            import ray
            ray.shutdown()


class PreTrainedLLM(ModelBase):
    def __init__(self, cfg_path: str, **kwargs):
        cfg = load_auto(cfg_path)

        super().__init__(cfg['model']['model_name_or_path'], cfg['model']['model_type'],
                         cfg['model']['max_output_length'], cfg['model']['use_vllm'], 
                         cfg['model']['prompt_system'], **kwargs)


class StruXGPTBase(ModelBase):
    def __init__(self, cfg_path: str, **kwargs):
        cfg = load_auto(cfg_path)

        self.prompt_system_path: str = cfg['model']['prompt_system_path']
        self.prompt_template_path: str = cfg['model']['prompt_template_path']
        self.load_prompt_template()

        self.sent_token_spacy: bool = cfg['data']['sent_token_spacy']
        self.build_sent_tokenizer()

        self.scfg = StructConfig(**cfg['struct_keys'])

        self.model_name_or_path = kwargs.get('model_name_or_path', cfg['model']['model_name_or_path'])

        super().__init__(self.model_name_or_path, cfg['model']['model_type'],
                         cfg['model']['max_output_length'], cfg['model']['use_vllm'], 
                         self.prompt_system, **kwargs)

    def __call__(self, text: str, return_json=True, **kwargs):
        raise NotImplementedError
    
    def batch_forward(self, text_list: List[str], return_json=True, **kwargs):
        raise NotImplementedError
    
    def prepare_prompt(self, text: str, **kwargs):
        raise NotImplementedError

    def build_sent_tokenizer(self):
        import spacy
        from src.utils.str import split_to_sentence as split_to_sentence_en
        from src.utils.str import split_to_sentence_zh

        if self.sent_token_spacy:
            self.sent_tokenizer_en = spacy.load('en_core_web_sm')
            self.sent_tokenizer_zh = spacy.load('zh_core_web_sm')
        else:
            self.sent_tokenizer_en = split_to_sentence_en
            self.sent_tokenizer_zh = split_to_sentence_zh

    def split_to_sentence(self, text: str) -> List[str]:
        sent_tokenizer = self.sent_tokenizer_zh if is_zh(text) else self.sent_tokenizer_en
        res = sent_tokenizer(text)

        return [s.text for s in res.sents] if self.sent_token_spacy else res

    def load_prompt_template(self):
        template = load_file(self.prompt_template_path)
        ## compatible for string-format and JSON-format in prompt template
        json_fomatter = re.findall('```json\n([^`]*)\n```', template)
        json_mapping = {k: f'##JSON{i}' for i, k in enumerate(json_fomatter)}
        for src, dst in json_mapping.items():
            template = template.replace(src, dst)
        
        self.prompt_template = template
        self.json_mapping = json_mapping

        self.prompt_system = load_file(self.prompt_system_path)

    def map_prompt_template(self, **kwargs):
        prompt = self.prompt_template.format(**kwargs)
        for src, dst in self.json_mapping.items():
            prompt = prompt.replace(dst, src)

        return prompt

    def count_token(self, text: str):
        return len(self.tokenizer.encode(text, truncation=False))

    def chunk_content(self, content: Union[List[str], str], 
                      max_length=1024, prefix='chunk', 
                      force_chunk=False) -> List[Dict[str, str]]:
        chunk_list_total = []
        if isinstance(content, list):
            for di, _content in enumerate(content):
                if is_empty(_content):
                    continue
                chunk_list_total.extend(
                    self.chunk_content(_content, max_length=max_length, 
                                       prefix=f'{prefix}_{di}',
                                       force_chunk=force_chunk)
                )
        else:
            cur_chunk_list, cur_tokens = [], 0
            for para in content.splitlines():
                if is_empty(para):
                    continue
                tokens = self.count_token(para) + 1
                if tokens > max_length * 1.0 and force_chunk:  # TODO: force_chunk
                    # print(f'Warning: long paragraph with {tokens} tokens.')
                    sentences = self.split_to_sentence(para)
                    tmp_sent_list = []
                    for sent in sentences:
                        sent_token = self.count_token(sent) + 1
                        # assume a sentence would not exceed `max_length`
                        if sent_token + cur_tokens > max_length:
                            cur_chunk_list.append(' '.join(tmp_sent_list))
                            chunk_list_total.append({
                                'idx': f'{prefix}_{len(chunk_list_total)}', 'data': '\n'.join(cur_chunk_list)
                            })
                            cur_chunk_list, cur_tokens, tmp_sent_list = [], 0, []
                        
                        tmp_sent_list.append(sent)
                        cur_tokens += sent_token
                    
                    if len(tmp_sent_list):
                        cur_chunk_list.append(' '.join(tmp_sent_list))

                    continue  # avoid redundancy

                if not force_chunk:
                    if (tokens + cur_tokens > max_length) and (tokens < 64 or cur_tokens > max_length) and len(cur_chunk_list):
                        chunk_list_total.append({
                            'idx': f'{prefix}_{len(chunk_list_total)}', 'data': '\n'.join(cur_chunk_list)
                        })
                        cur_chunk_list, cur_tokens = [], 0
                else:
                    if tokens + cur_tokens > max_length:
                        assert len(cur_chunk_list)
                        title_candidate = cur_chunk_list[-1]
                        last_chunk_is_title = title_candidate[-1] not in PUNCTUATION and \
                                              self.count_token(title_candidate) < 32
                        if last_chunk_is_title:
                            cur_chunk_list = cur_chunk_list[:-1]
                        if len(cur_chunk_list):
                            chunk_list_total.append({
                                'idx': f'{prefix}_{len(chunk_list_total)}', 'data': '\n'.join(cur_chunk_list)
                            })
                        cur_chunk_list, cur_tokens = [], 0
                        if last_chunk_is_title:
                            cur_chunk_list.append(title_candidate)
                            cur_tokens = self.count_token(title_candidate) + 1

                cur_chunk_list.append(para)
                cur_tokens += tokens

            if len(cur_chunk_list):
                chunk_list_total.append({
                    'idx': f'{prefix}_{len(chunk_list_total)}', 'data': '\n'.join(cur_chunk_list)
                })
        
        return chunk_list_total