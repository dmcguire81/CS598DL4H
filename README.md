# CS598DL4H
Project for CS 598 Deep Learning for Healthcare

## Local Setup

### Submodules

```sh
git submodule init
git submodule update
```

### Virtual Environments

#### CS598DL4h

```sh
pyenv local 3.7.11
python3.7 -m venv env
env/bin/pip install --upgrade pip
env/bin/pip install -r requirements.txt -r dev-requirements.txt
```

Use `env/bin/python` as the kernel for [MIMIC_III.ipynb](./MIMIC_III.ipynb).

#### caml-mimic

```sh
pyenv local 3.7.11
python3.7 -m venv env
env/bin/pip install --upgrade pip
env/bin/pip install -r requirements.txt
```

Use `caml-mimic/env/bin/python` as the kernel for [caml-mimic/notebooks/dataproc_mimic_III.ipynb](caml-mimic/notebooks/dataproc_mimic_III.ipynb).

#### Explainable-Automated-Medical-Coding

```sh
pyenv local 3.7.11
python3.7 -m venv env
env/bin/pip install --upgrade pip
env/bin/pip install -r requirements.txt
```

Use `Explainable-Automated-Medical-Coding/env/bin/python` as the kernel for [Explainable-Automated-Medical-Coding/HLAN/demo_HLAN_viz.ipynb](Explainable-Automated-Medical-Coding/HLAN/demo_HLAN_viz.ipynb).

## Google Colab Setup

See [Setup.ipynb](./Setup.ipynb) for Google Colab-only set up steps. Also, relevant Colab-only header sections in each notebook reference this set up.

## Repro Steps

### Prerequisites / Demo

[MIMIC_III.ipynb](./MIMIC_III.ipynb)

### Training

Examples detailed in [Explainable-Automated-Medical-Coding/README.md](./Explainable-Automated-Medical-Coding/README.md).

```sh
cd Explainable-Automated-Medical-Coding/HLAN/
source ../env/bin/activate
```

#### MIMIC-III Top 50

Currently working off of `Explainable-Automated-Medical-Coding/datasets/mimiciii_*_50_th0.txt`.

##### Original

```sh
cd Explainable-Automated-Medical-Coding/HLAN/
../env/bin/python HAN_train.py \
    --dataset mimic3-ds-50 \
    --batch_size 32 \
    --per_label_attention=True \
    --per_label_sent_only=False \
    --num_epochs 100 \
    --report_rand_pred=False \
    --running_times 1 \
    --early_stop_lr 0.00002 \
    --remove_ckpts_before_train=False \
    --use_label_embedding=True \
    --ckpt_dir ../checkpoints/checkpoint_HAN_50_per_label_bs32_LE/ \
    --use_sent_split_padded_version=False \
    --marking_id 50-hlan \
    --gpu=True  # Colab only
```

##### Clone

```sh
# Most of the flags and options are initially hard-coded to the below values
env/bin/python HLAN/HAN_train.py \
    --dataset 'caml-mimic/mimicdata/mimic3/*_50.csv' \
    --batch_size 32 \
    --per_label_attention=True \
    --per_label_sent_only=False \
    --num_epochs=100 \
    --report_rand_pred=False \
    --running_times=1 \
    --early_stop_lr=0.00002 \
    --remove_ckpts_before_train=False \
    --use_label_embedding=True \
    --ckpt_dir Explainable-Automated-Medical-Coding/checkpoints/checkpoint_HAN_50_per_label_bs32_LE \
    --use_sent_split_padded_version=False \
    --marking_id mimic3-ds-50 \
    --gpu=True \
    --log_dir logs \
    --word2vec_model_path Explainable-Automated-Medical-Coding/embeddings/processed_full.w2v \
    --label_embedding_model_path Explainable-Automated-Medical-Coding/embeddings/code-emb-mimic3-tr-400.model \
    --label_embedding_model_path_per_label Explainable-Automated-Medical-Coding/embeddings/code-emb-mimic3-tr-200.model
```

#### MIMIC-III COVID-19 Shielding

Needs preprocessing to extract only Admissions IDs from admissions containing COVID-19 related ICD-9 codes, derived from COVID-19 related ICD-10 codes in `./spl-icd10-opcs4-disease-groups-v2.0.csv` and from an ICD-10-to-ICD-9 mapping in `./masterb8.csv`. This process needs to generate `CSV` files akin to `caml-mimic/mimicdata/mimic3/*_50.csv` to be converted by `csv_to_text.py` to `Explainable-Automated-Medical-Coding/datasets/mimiciii_*_full_th_50_covid_shielding.txt`, the files the training expects.

##### Original

```sh
cd Explainable-Automated-Medical-Coding/HLAN/
../env/bin/python HAN_train.py \
    --dataset mimic3-ds-shielding-th50 \
    --batch_size 32 \
    --per_label_attention=True \
    --per_label_sent_only=False \
    --num_epochs 100 \
    --report_rand_pred=False \
    --running_times 1 \
    --early_stop_lr 0.00002 \
    --remove_ckpts_before_train=False \
    --use_label_embedding=True \
    --ckpt_dir ../checkpoints/checkpoint_HAN_shielding_per_label_bs32_LE/ \
    --use_sent_split_padded_version=False \
    --marking_id shielding-hlan \
    --gpu=True  # Colab only
```