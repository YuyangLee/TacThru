#!/usr/bin/bash

task=pull_tissue # One of: pick_bottle, pull_tissue, sort_bolt, hang_scissors, insert_cap

tac_active_keys="[tacthru_l_rgb,tacthru_l_marker]" # TacThru w/ marker deviations
obs_tag="tt_m"
exp_tag="run"

uv run scripts/train.py --config-name=train_tf exp_name=tf-$obs_tag-$exp_tag task=$task tac_active_keys=$tac_active_keys
