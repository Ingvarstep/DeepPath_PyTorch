#!/bin/bash

relation=$1
python3 train_policy_supervised_learning.py $relation
python3 train_policy_reinforcement_learning.py $relation retrain
python3 train_policy_reinforcement_learning.py $relation test


