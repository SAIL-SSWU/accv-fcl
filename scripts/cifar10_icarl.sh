#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
--wandb=0 \
--dataset=cifar10 \
--method=icarl \
--tasks=5 \
--beta=0.5 \
--num_users=10 \
--frac=1.0 \
--com_round=20 \
--local_ep=5 \
--local_bs=128 \
--memory_size=50 \
--increment=2