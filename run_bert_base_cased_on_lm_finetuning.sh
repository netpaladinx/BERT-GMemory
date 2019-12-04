#!/bin/bash

TRAIN_FILE=./datasets/wikitext-103-raw/wiki.train.raw
TEST_FILE=./datasets/wikitext-103-raw/wiki.test.raw
TASK_NAME=LM_FineTuning_wiki103
OUTPUT_DIR=./output/$TASK_NAME

python run_lm_finetuning.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --do_train \
  --do_eval \
  --train_data_file $TRAIN_FILE \
  --eval_data_file $TEST_FILE \
  --mlm \
  --per_gpu_train_batch_size 4 \
  --per_gpu_eval_batch_size 4 \
  --num_train_epochs 5.0 \
  --output_dir $OUTPUT_DIR