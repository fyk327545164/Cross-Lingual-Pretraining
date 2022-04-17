#!/bin/sh

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yfeng55/anaconda3/lib/

for ratio in 5 10 15 20 25 30 50
do
  python3 run_${1}.py --ratio=$ratio --mode=$2
done