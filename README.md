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
env/bin/pip install -r requirements.txt
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

[MIMIC_III.ipynb](./MIMIC_III.ipynb)