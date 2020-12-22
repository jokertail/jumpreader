#!/usr/bin/env bash
#python run.py --generate_emb --generate_mode "train" --batch-size 32 --test-batch-size 64 --model-type "jump" --model-name "jump-2-10-10" --train-file 'train' --dev-file 'dev' --output-dir 'output/long' --data-dir 'data/quasar/long' --para-num 6 --overwrite_cache

python run.py --do-train --batch-size 2 --test-batch-size 6 --model-type "jump" --model-name "jump-N1-K10"  --N 1 --K 10 --train-file 'train' --dev-file 'dev' --test-file 'test' --output-dir 'output/long' --data-dir 'data/quasar/long' --para-num 6  --init-step 100000 --num-epochs 20 --logging-steps 50 --learning-rate 0.01 --warmup-steps 500 --ckpt-file jump-N2-K10_checkpoint-190000
#--load-ckpt
