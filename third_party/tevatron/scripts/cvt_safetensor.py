import torch
from safetensors.torch import save_file

# data_dir = '/home/chengan.lk/liuk/cache/huggingface/hub/models--castorini--repllama-v1-7b-lora-passage/snapshots/53f3ec543eafdc511092ade9de3f8b5a00155a3e'
# src_file = f'{data_dir}/adapter_model.bin'
# dst_file = f'{data_dir}/adapter_model.safetensors'

# file = torch.load(src_file)
# save_file(file, dst_file)

import pickle
import numpy as np

def pickle_load(path):
    with open(path, 'rb') as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup

data = pickle_load('emb_beir/scifact/corpus_scifact.pkl')
print(data)