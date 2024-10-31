## Free-Form Evaluation on LongBench

Make sure you are in the **root path** of this repo, and run the following commands.

```bash
cd /path/to/StruXGPT
```

### Data Preparation

First, download the data folder from [this link](https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip) and unzip it to `third_party/LongBench/data`.
We take the passages for knowledge injection, generate free-form QA examples for instruction alignment, and evaluate the obtained models with LongBench's original questions (without passages in the context).

Second, run the following command:

```bash
python examples/StructTuning/LongBench/preprocess.py
```

The training data will be automatically generated:
- CPT: `third_party/LLaMA-Factory/data/longbench/longbench_psg_train_cpt.json`
- SCPT (Ours): `third_party/LLaMA-Factory/data/longbench/longbench_psg_train_scpt.json`
- SFT: `third_party/LLaMA-Factory/data/longbench/longbench_psg_train_sft.json`
- SSFT (Ours): `third_party/LLaMA-Factory/data/longbench/longbench_psg_train_ssft.json`

Then, move to the `third_party/LLaMA-Factory` toolkit to turn a pre-trained model (e.g., llama2-7b) into a domain-specific expert:

```bash
cd third_party/LLaMA-Factory
chmod a+x bash/*.sh
```

### Continual Pre-Training

- Train a baseline model with Vanilla-CPT:
```bash
CPT_TYPE=vanilla MODEL_TYPE=llama2 bash ./bash/train_longbench_cpt.sh
```

- Train a target model with our Struct-CPT (SCPT):
```bash
CPT_TYPE=struct MODEL_TYPE=llama2 bash ./bash/train_longbench_cpt.sh
```

### Supervised Fine-Tuning

- Train a baseline model with Vanilla-SFT:
```bash
SFT_TYPE=vanilla MODEL_TYPE=llama2 bash ./bash/train_longbench_sft.sh
```

- Train a target model with our Struct-SFT (SSFT):
```bash
SFT_TYPE=struct MODEL_TYPE=llama2 bash ./bash/train_longbench_sft.sh
```

### Evaluation

After continual pre-training and supervised fine-tuning, go back to the root path of this repo, and run the following code for evaluation:

```bash
cd /path/to/StruXGPT
python examples/StructTuning/LongBench/evaluate.py --model config/longbench_llama2_7b.yaml
```

**Note**: Change the `model_name_or_path` in `config/longbench_llama2_7b.yaml` to evaluate different models.