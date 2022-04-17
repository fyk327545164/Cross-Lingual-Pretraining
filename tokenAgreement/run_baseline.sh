#!/bin/sh

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yfeng55/anaconda3/lib/

for task in pos ner qa xnli
do
  python3 run_${task}.py --ratio=0 --mode=code
done