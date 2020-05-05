#!/bin/bash

# To save adversaria images

# === Set parameters ===
normal='false'
n_rows=8
batch_size=1
set_idx='true'
indices=(1 10 100)
attack_method='DeepFool' # FGSM | PGD | CW | DeepFool
dataset='cifar10' #cifar10 | cifar100

# run main module
if [ "$normal"="true" ]; then
    echo "Start saving the normal images"
else
    echo "Start saving the adversarial images made by $attack_method"
fi

python visualization.py --vis-n-rows $n_rows --vis-batch-size $batch_size\
 --vis-set-idx $set_idx --vis-indices ${indices[@]} --dataset $dataset\
 --attack-name $attack_method --vis-normal $normal
