# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

import json

import datasets

_CITATION = '''
@inproceedings{
    thakur2021beir,
    title={{BEIR}: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
    author={Nandan Thakur and Nils Reimers and Andreas R{\"u}ckl{\'e} and Abhishek Srivastava and Iryna Gurevych},
    booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
    year={2021},
    url={https://openreview.net/forum?id=wCu6T5xFjeJ}
}
'''

all_data = [
    'arguana',
    'climate-fever',
    'cqadupstack-android',
    'cqadupstack-english',
    'cqadupstack-gaming',
    'cqadupstack-gis',
    'cqadupstack-mathematica',
    'cqadupstack-physics',
    'cqadupstack-programmers',
    'cqadupstack-stats',
    'cqadupstack-tex',
    'cqadupstack-unix',
    'cqadupstack-webmasters',
    'cqadupstack-wordpress',
    'dbpedia-entity',
    'fever',
    'fiqa',
    'hotpotqa',
    'nfcorpus',
    'quora',
    'scidocs',
    'scifact',
    'trec-covid',
    'webis-touche2020',
    'nq'
]

_DESCRIPTION = 'dataset load script for BEIR corpus'

_DATASET_URLS = {
    data: {
        # 'train': f'https://huggingface.co/datasets/Tevatron/beir-corpus/resolve/main/{data}.jsonl.gz',
        'train': f'data/struct_s4_h1_f2/{data}.jsonl',  #_remap
    } for data in all_data
}


class BeirCorpus(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            version=datasets.Version('1.1.0'),
            name=data, 
            description=f'BEIR dataset corpus {data}.'
        ) for data in all_data
    ]

    def _info(self):
        features = datasets.Features({
            'docid': datasets.Value('string'), 
            'title': datasets.Value('string'),
            'text': datasets.Value('string'),
        })

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage='https://github.com/beir-cellar/beir',
            # License for the dataset if available
            license='',
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data = self.config.name
        downloaded_files = dl_manager.download_and_extract(_DATASET_URLS[data])

        splits = [
            datasets.SplitGenerator(
                name='train',
                gen_kwargs={
                    'filepath': downloaded_files['train'],
                },
            ),
        ]
        return splits

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                yield data['docid'], data