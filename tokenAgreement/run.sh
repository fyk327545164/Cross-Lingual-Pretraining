#!/bin/sh

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yfeng55/anaconda3/lib/

for ratio in 0.05 0.1 0.15 0.2 0.25 0.3 0.5
do
  python3 run_${1}.py --ratio=$ratio --mode=$2
done