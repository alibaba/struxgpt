# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import jsonlines
import yaml
from tqdm import tqdm


def load_json(p):
    with open(p, 'r') as f:
        data = json.load(f)
    return data


def load_jsonl(p, show_progress=False):
    with open(p, 'r') as f:
        if show_progress:
            f = tqdm(f, desc='Loading jsonl')
        data = [json.loads(line) for line in f]
    return data


def load_file(p, splitline=False):
    with open(p, 'r') as f:
        content = f.read()
    if splitline:
        content = content.split('\n')
    return content


def load_yaml(p):
    with open(p, 'r') as file:
        data = yaml.safe_load(file)
    return data


def load_auto(p: str, show_progress=False):
    ext = p.split('.')[-1]
    if ext == 'jsonl':
        return load_jsonl(p, show_progress=show_progress)
    elif ext in ['json', 'yaml']:
        return eval(f'load_{ext}')(p)
    else:
        return load_file(p)


def write_jsonl(p, data, **kwargs):
    with jsonlines.open(p, 'w') as f:
        f.write_all(data, **kwargs)


def write_json(p, data, **kwargs):
    with open(p, 'w+') as f:
        json.dump(data, f, 
                  indent=kwargs.get('indent', 4), 
                  ensure_ascii=kwargs.get('ensure_ascii', False), **kwargs)


def write_file(p, text, **kwargs):
    with open(p, 'w+') as f:
        f.write(text, **kwargs)


def write_yaml(p, data, **kwargs):
    sort_keys = kwargs.get('sort_keys', False)
    indent = kwargs.get('indent', 2)
    default_flow_style = kwargs.get('default_flow_style', False)
    allow_unicode = kwargs.get('allow_unicode', True)
    content = yaml.safe_dump(data, 
                             sort_keys=sort_keys, 
                             indent=indent,
                             default_flow_style=default_flow_style, 
                             allow_unicode=allow_unicode)
    with open(p, 'w+') as f:
       f.write(content)


def write_auto(p: str, data, disp=False, **kwargs):
    if disp:
        print(p, len(data))
    ext = p.split('.')[-1]
    if ext in ['json', 'jsonl', 'yaml']:
        eval(f'write_{ext}')(p, data, **kwargs)
    else:
        write_file(p, data)


def path_suffix_auto(p: str, suffix: str):
    ext = p.split('.')[-1]
    return p.replace(f'.{ext}', f'{suffix}.{ext}')