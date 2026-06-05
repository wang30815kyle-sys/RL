# Server Run Guide

This repository contains the code and model snapshots needed to run the
60% random / 40% snapshot-pool continuation training.

## Setup

```bash
git clone https://github.com/wang30815kyle-sys/RL.git
cd RL

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Conda alternative:

```bash
conda env create -f environment.yml
conda activate mlarena
pip install -r requirements.txt
```

## Check Files

```bash
python -m py_compile train.py model.py agent.py evaluate.py pvp_evaluate.py

ls -lh \
  model.zip \
  model_2949k_snapshot_pool_champion.zip \
  model_3457k_80random_specialist.zip \
  model_2441k_episode_fixed_champion.zip \
  model_1933k_50_50_win33.zip \
  model_1425k_white_primary_before_color_mix.zip \
  model_500k_reward_prev.zip
```

`model.zip` is the starting checkpoint for training.

## Train

Foreground:

```bash
python -u train.py
```

Background:

```bash
nohup python -u train.py > train_60random_40pool_server.log 2> train_60random_40pool_server.err.log &
echo $!
```

Monitor:

```bash
tail -f train_60random_40pool_server.log
tail -f train_60random_40pool_server.err.log
```

After training:

```bash
cp model.zip model_after_60random_40pool_500k.zip
```

## Evaluate

Random opponent:

```bash
python evaluate.py --games 1000 --model model.zip --seed 12345
python evaluate.py --games 1000 --model model.zip --seed 67890
```

PvP:

```bash
python pvp_evaluate.py --model-a model.zip --model-b model_2949k_snapshot_pool_champion.zip --games 500 --seed 12345 --stochastic
python pvp_evaluate.py --model-a model.zip --model-b model_3457k_80random_specialist.zip --games 500 --seed 12345 --stochastic
python pvp_evaluate.py --model-a model.zip --model-b model_500k_reward_prev.zip --games 500 --seed 12345 --stochastic
```

Quick deterministic smoke:

```bash
python pvp_evaluate.py --model-a model.zip --model-b model_2949k_snapshot_pool_champion.zip --games 2 --seed 12345
```
