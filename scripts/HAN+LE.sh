#!/bin/bash

env/bin/python HLAN/HAN_train.py \
    --dataset 'caml-mimic/mimicdata/mimic3/*_50.csv' \
    --batch_size 32 \
    --per_label_attention=False \
    --per_label_sent_only=False \
    --num_epochs=100 \
    --report_rand_pred=False \
    --running_times=1 \
    --early_stop_lr=0.00002 \
    --remove_ckpts_before_train=False \
    --use_label_embedding=True \
    --ckpt_dir Explainable-Automated-Medical-Coding/checkpoints/no_checkpoint \
    --use_sent_split_padded_version=False \
    --marking_id mimic3-ds-50 \
    --gpu=True \
    --log_dir logs/HAN+LE \
    --word2vec_model_path Explainable-Automated-Medical-Coding/embeddings/processed_full.w2v \
    --label_embedding_model_path Explainable-Automated-Medical-Coding/embeddings/code-emb-mimic3-tr-400.model \
    --label_embedding_model_path_per_label Explainable-Automated-Medical-Coding/embeddings/code-emb-mimic3-tr-200.model