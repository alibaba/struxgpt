ckpt=$1
# tokenizer=bert-base-uncased
tokenizer=${ckpt}

for suffix in "" "-struct"; do
  embedding_dir=embeddings/emb_beir/${ckpt}${suffix}
  mkdir -p ${embedding_dir}

  datasets=" nfcorpus fiqa arguana scidocs scifact "
  for dataset in ${datasets}; do
    python -m tevatron.driver.encode \
      --output_dir=temp \
      --model_name_or_path ${ckpt} \
      --tokenizer_name ${tokenizer} \
      --fp16 \
      --per_device_eval_batch_size 64 \
      --p_max_len 512 \
      --dataset_name ./datasets/beir-corpus${suffix}:${dataset} \
      --encoded_save_path ${embedding_dir}/corpus_${dataset}.pkl \
      --encode_num_shard 1 \
      --encode_shard_index 0

    python -m tevatron.driver.encode \
      --output_dir=temp \
      --model_name_or_path ${ckpt} \
      --tokenizer_name ${tokenizer} \
      --fp16 \
      --per_device_eval_batch_size 64 \
      --dataset_name ./datasets/beir:${dataset}/test \
      --encoded_save_path ${embedding_dir}/query_${dataset}.pkl \
      --q_max_len 512 \
      --encode_is_qry

    set -f && OMP_NUM_THREADS=12 python -m tevatron.faiss_retriever \
        --query_reps ${embedding_dir}/query_${dataset}.pkl \
        --passage_reps ${embedding_dir}/corpus_${dataset}.pkl \
        --depth 100 \
        --batch_size 64 \
        --save_text \
        --save_ranking_to ${embedding_dir}/rank.${dataset}.txt

    python -m tevatron.utils.format.convert_result_to_trec --input ${embedding_dir}/rank.${dataset}.txt \
                                                          --output ${embedding_dir}/rank.${dataset}.trec \
                                                          --remove_query
    echo "####${dataset}${suffix}####"
    python -m pyserini.eval.trec_eval -c -mrecall.100 -mndcg_cut.10 beir-v1.0.0-${dataset}-test ${embedding_dir}/rank.${dataset}.trec

    done
done