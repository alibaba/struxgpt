# Enhancing Large Language Models via Structurization

This repo integrates a series of research projects focused on enhancing cognition abilities or injecting domain knowledge for LLMs via *structurization*:

* [Enhancing LLM’s Cognition via Structurization](https://arxiv.org/abs/2407.16434), NeurIPS-24.
* [Educating LLMs like Human Students: Structure-aware Injection of Domain Knowledge](https://arxiv.org/abs/2407.16724), ArXiv'24.

## Updates

- **27.08.2024**: Upload the codebase. Data and weights are coming.


## How to Run
Install the dependencies:

```
pip install -r requirements.txt
```

Click a paper below to see the detailed instructions on how to run the code in `examples/*` to reproduce the results.

* [Enhancing LLM’s Cognition via Structurization](examples/StruXGPT/README.md)
* [Educating LLMs like Human Students: Structure-aware Injection of Domain Knowledge](examples/StructTuning/README.md)


## Citation
If you use this code in your research, please kindly cite the following papers

```
@article{liu2024enhancing,
      title={Enhancing LLM's Cognition via Structurization}, 
      author={Liu, Kai and Fu, Zhihang and Chen, Chao and Zhang, Wei and Jiang, Rongxin and Zhou, Fan and Chen, Yaowu and Wu, Yue and Ye, Jieping},
      year={2024},
      eprint={2407.16434},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.16434}, 
}

@article{liu2024educating,
      title={Educating LLMs like Human Students: Structure-aware Injection of Domain Knowledge}, 
      author={Liu, Kai and Chen, Ze and Fu, Zhihang and Jiang, Rongxin and Zhou, Fan and Chen, Yaowu and Wu, Yue and Ye, Jieping},
      year={2024},
      eprint={2407.16724},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.16724}, 
}
```
