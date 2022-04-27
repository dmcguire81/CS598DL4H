#!/bin/bash

env/bin/python HLAN/HAN_train.py \
    --dataset 'caml-mimic/mimicdata/mimic3/*_50.csv' \
    --batch_size 32 \
    --per_label_attention=True \
    --per_label_sent_only=False \
    --num_epochs=100 \
    --early_stop_lr=0.00002 \
    --remove_ckpts_before_train=False \
    --use_label_embedding=False \
    --ckpt_dir checkpoints/HLAN/ \
    --log_dir logs/HLAN \
    --word2vec_model_path Explainable-Automated-Medical-Coding/embeddings/processed_full.w2v \
    --label_embedding_model_path Explainable-Automated-Medical-Coding/embeddings/code-emb-mimic3-tr-400.model \
    --label_embedding_model_path_per_label Explainable-Automated-Medical-Coding/embeddings/code-emb-mimic3-tr-200.model
