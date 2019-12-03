#!/bin/bash

TRAIN_FILE=./datasets/wikitext-2/wiki.train.raw
TEST_FILE=./datasets/wikitext-2/wiki.test.raw
TASK_NAME=/LM_FineTuning

python run_lm_finetuning.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --do_train \
  --do_eval \
  --train_data_file $TRAIN_FILE \
  --eval_data_file $TEST_FILE \
  --mlm \
  --per_gpu_train_batch_size 32 \
  --num_train_epochs 1.0 \
  --output_dir /tmp/$TASK_NAME/