#!/bin/bash

# Variables
export PRETRAINED_PATH=experiments/pretrain_lr1e-3_epoch5_wd0.0_20260309_055220/pretrained_model.pt
export EXP_DIR=experiments/finetune_pretrain_epoch5

export EPOCHS=2
export CHECKPOINT_EVERY=0.1

# GRPO (RL without KL regularization)
python scripts/finetune.py --pretrained-model $PRETRAINED_PATH --exp-dir $EXP_DIR --epochs $EPOCHS --checkpoint-every $CHECKPOINT_EVERY --method grpo --wandb --wandb-name grpo_epoch${EPOCHS}

# GRPO with KL regularization
python scripts/finetune.py --pretrained-model $PRETRAINED_PATH --exp-dir $EXP_DIR --epochs $EPOCHS --checkpoint-every $CHECKPOINT_EVERY --method grpo_kl --wandb --wandb-name grpo_kl_epoch${EPOCHS}

# SFT variants
python scripts/finetune.py --pretrained-model $PRETRAINED_PATH --exp-dir $EXP_DIR --epochs $EPOCHS --checkpoint-every $CHECKPOINT_EVERY --method sft1 --wandb --wandb-name sft1_epoch${EPOCHS}
python scripts/finetune.py --pretrained-model $PRETRAINED_PATH --exp-dir $EXP_DIR --epochs $EPOCHS --checkpoint-every $CHECKPOINT_EVERY --method sft2 --wandb --wandb-name sft2_epoch${EPOCHS}
python scripts/finetune.py --pretrained-model $PRETRAINED_PATH --exp-dir $EXP_DIR --epochs $EPOCHS --checkpoint-every $CHECKPOINT_EVERY --method oracle --wandb --wandb-name oracle_epoch${EPOCHS}
