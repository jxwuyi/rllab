#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES='' python3 trpo_singleagent_discr_comm.py \
    --desc 4x4-empty --iter 1001 --pathlen 10 --maxsteps 10000 \
    --n_goal 2 --gamma 0.8 \
    --logdir default --store_gap 20 \
    --baseline mlp
    # --opt-msg --agent-escape
