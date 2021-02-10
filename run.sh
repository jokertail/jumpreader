#!/usr/bin/env bash
#python run.py --generate_emb --generate_mode "train" --batch-size 32 --test-batch-size 64 --model-type "jump" --model-name "jump-2-10-10" --train-file 'train' --dev-file 'dev' --output-dir 'output/long' --data-dir 'data/quasar/long' --para-num 6 --overwrite_cache

python run.py --do-test --batch-size 8 --test-batch-size 12 --model-type "jump" --model-name "test-N2K10"  --N 2 --K 10 --train-file 'train' --dev-file 'dev' --test-file 'test' --output-dir 'output/long' --data-dir 'data/quasar/long' --para-num 6  --init-step 220000 --num-epochs 5 --logging-steps 50 --learning-rate 0.01 --warmup-steps 500 --ckpt-file jump-N1-K10-noQ_checkpoint-222500 --load-ckpt
#"jump-N1-K10-noQ"
