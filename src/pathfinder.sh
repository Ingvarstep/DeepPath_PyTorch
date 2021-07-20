#!/bin/bash
cd /content/DeepPath_PyTorch-master/src
relation=$1
path=$2
python3 train_policy_supervised_learning.py $relation $path
python3 train_policy_reinforcement_learning.py $relation retrain $path
python3 train_policy_reinforcement_learning.py $relation test $path





