## Multi-Choice Evaluation on MMedBench

Make sure you are in the **root path** of this repo, and run the following commands.

```bash
cd /path/to/StruXGPT
```

### Data Preparation

First, create the directory to organize the training and testing data:

```bash
mkdir -p data/tune/MMedBench
```

The dataset for medicine knowledge injection comes from two resouces:
- [textbooks](https://github.com/jind11/MedQA): around 30M tokens from medicine textbook in English and Chinese. Download from [this link](https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view?usp=sharing) and put it into `data/tune/MMedBench/textbooks`;
- [MMedC](https://github.com/MAGIC-AI4Med/MMedLM): developed with [MMedBench](https://github.com/MAGIC-AI4Med/MMedLM), which contains 25.5B tokens in total. Download the [dataset](https://huggingface.co/datasets/Henrychur/MMedC) and put it into `data/tune/MMedBench/MMedC`. We only use a part of this corpus with high quality textbooks as a supplementary in the remaining 4 langauges.

Download the [MMedBench](https://huggingface.co/datasets/Henrychur/MMedBench) dataset at `data/tune/MMedBench/MMedBench` for instruction alignment and model evaluation, which involves multi-choice diagnoses QA pairs across 6 languages (45K for training and 8.5K for testing). 

Then, run the following command:

```bash
python examples/StructTuning/MMedBench/preprocess.py
```

The training data will be automatically generated:
- CPT: `third_party/LLaMA-Factory/data/mmedbench/mmedbench_corpus_train_cpt.json`
- SCPT (Ours): `third_party/LLaMA-Factory/data/mmedbench/mmedbench_corpus_train_scpt.json`
- SFT: `third_party/LLaMA-Factory/data/mmedbench/mmedbench_instruct_train_sft.json`
- SSFT (Ours): `third_party/LLaMA-Factory/data/mmedbench/mmedbench_instruct_train_ssft.json`

Then, move to the `third_party/LLaMA-Factory` toolkit to turn a pre-trained model (e.g., llama2-7b) into a domain-specific expert:

```bash
cd third_party/LLaMA-Factory
chmod a+x bash/*.sh
```

### Continual Pre-Training

- Train a baseline model with Vanilla-CPT:
```bash
CPT_TYPE=vanilla MODEL_TYPE=llama2 bash ./bash/train_mmedbench_cpt.sh
```

- Train a target model with our Struct-CPT (SCPT):
```bash
CPT_TYPE=struct MODEL_TYPE=llama2 bash ./bash/train_mmedbench_cpt.sh
```

### Supervised Fine-Tuning

- Train a baseline model with Vanilla-SFT:
```bash
SFT_TYPE=vanilla MODEL_TYPE=llama2 bash ./bash/train_mmedbench_sft.sh
```

- Train a target model with our Struct-SFT (SSFT):
```bash
SFT_TYPE=struct MODEL_TYPE=llama2 bash ./bash/train_mmedbench_sft.sh
```

### Evaluation

After continual pre-training and supervised fine-tuning, go back to the root path of this repo, and run the following code for evaluation:

```bash
cd /path/to/StruXGPT
python examples/StructTuning/MMedBench/evaluate.py --model config/mmedbench_llama2_7b.yaml
```

**Note**: Change the `model_name_or_path` in `config/mmedbench_llama2_7b.yaml` to evaluate different models.