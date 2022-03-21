#!/bin/sh

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yfeng55/anaconda3/lib/

for lg in hi ru zh tr de
do
  for ratio in 0 5 10 15 20 25 30 50
  do
    python3 run_finetune_new.py --ratio=$ratio --lg=$lg
  done
done