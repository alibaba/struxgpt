#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH # using nvcc in cuda-dir, for cpu_adam compile
export PYTHONPATH='.'

max_seq_len="${MAX_SEQ_LEN:-4096}"
num_epoch="${NUM_EPOCH:-2}"
global_batch_size="${GLOBAL_BATCH_SIZE:-128}"
batch_size_per_device="${BATCH_SIZE_PER_DEVICE:-2}"
ngpus="${NGPUS:-8}"
accum_steps=$((${global_batch_size}/${ngpus}/${batch_size_per_device}))

model_name_or_path="${MODEL_NAME_OR_PATH:-"./weights/Llama-2-7b-chat-hf"}"
model_type="${MODEL_TYPE:-"llama2"}"
cpt_type="${CPT_TYPE:-"struct"}"
if [[ $cpt_type == "struct" ]]; then
    template="${model_type}_plain"
    stage="sft"
else
    template="vanilla"
    stage="pt"
fi

learning_rate="${LEARNING_RATE:-2e-5}"
min_learning_rate="${MIN_LEARNING_RATE:-2e-6}"
lr_scheduler_type="${LR_SCHEDULER_TYPE:-cosine}"
weight_decay="${WEIGHT_DECAY:-0.0}"

deepspeed="${DEEPSPEED:-"examples/full_multi_gpu/ds_z2_config.json"}"

dtype="${DTYPE:-bf16}"
if [[ $dtype == "fp16" ]]; then 
  fp16=true 
else 
  fp16=false 
fi
if [[ $dtype == "bf16" ]]; then 
  bf16=true 
else 
  bf16=false 
fi


deepspeed --num_gpus ${ngpus} src/train_bash.py \
    --deepspeed ${deepspeed} \
    --stage ${stage} \
    --do_train \
    --model_name_or_path ${model_name_or_path} \
    --trust_remote_code True \
    --dataset longbench_cpt_${cpt_type} \
    --dataset_dir ./data \
    --template ${template} \
    --finetuning_type full \
    --output_dir ./output/LongBench-CPT-${cpt_type}-${model_type} \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len ${max_seq_len} \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size ${batch_size_per_device} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${accum_steps} \
    --learning_rate ${learning_rate} \
    --min_learning_rate ${min_learning_rate} \
    --weight_decay ${weight_decay} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --logging_steps 10 \
    --warmup_steps 20 \
    --evaluation_strategy "no" \
    --num_train_epochs ${num_epoch} \
    --ddp_timeout 1800000 \
    --plot_loss \
    --gradient_checkpointing \
    --fp16 ${fp16} \
    --bf16 ${bf16}
