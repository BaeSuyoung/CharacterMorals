#!/bin/sh

python3 models/train.py \
--model_type bert \
--model_name_or_path bert-large-uncased \
--task_name AICA \
--do_eval \
--do_prediction \
--do_lower_case \
--data_dir data \
--max_seq_length 100 \
--per_gpu_eval_batch_size 8 \
--per_gpu_train_batch_size 8 \
--learning_rate 1e-5 \
--num_train_epochs 50 \
--output_dir output \
--do_train \
--logging_steps 500 \
--save_steps 500 \
--seed 42 \
--data_cache_dir cache \
--warmup_pct 0.1 \
--evaluate_during_training \
--save_total_limit 20 \
--patience 10 \
--overwrite_output_dir
