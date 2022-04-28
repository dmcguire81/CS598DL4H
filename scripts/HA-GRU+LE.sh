#!/bin/bash

python HLAN/HAN_train.py \
    --dataset 'caml-mimic/mimicdata/mimic3/*_50.csv' \
    --batch_size 32 \
    --per_label_attention=True \
    --per_label_sent_only=True \
    --num_epochs=100 \
    --early_stop_lr=0.00002 \
    --remove_ckpts_before_train=False \
    --use_label_embedding=True \
    --ckpt_dir checkpoints/HA-GRU+LE/ \
    --log_dir logs/HA-GRU+LE \
    --word2vec_model_path Explainable-Automated-Medical-Coding/embeddings/processed_full.w2v \
    --label_embedding_model_path Explainable-Automated-Medical-Coding/embeddings/code-emb-mimic3-tr-400.model \
    --label_embedding_model_path_per_label Explainable-Automated-Medical-Coding/embeddings/code-emb-mimic3-tr-200.model
