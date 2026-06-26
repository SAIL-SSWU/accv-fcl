#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
--wandb=0 \
--dataset=cifar100 \
--method=icarl \
--tasks=5 \
--beta=0.5 \
--num_users=20 \
--frac=0.4 \
--com_round=100 \
--local_ep=5 \
--local_bs=128 \
--memory_size=200 \
--increment=20 \