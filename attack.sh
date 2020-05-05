#!/bin/bash

# To create adversarial examples
# Available attack methods: [FGSM|PGD|CW|DeepFool]

# === Set parameters ===
# Hyper-parameters are set default value
# If ones want change the values, you can change below variables:
# For PGD: pgd-iters, pgd-eps, pgd-alpha, pgd-random-start
# For CW: cw-c, cw-kappa, cw-iters, cw-lr, cw-binary-search-steps
# For FGSM: fgsm-eps
# For DeepFool: deepFool-iters
# You can check the meaning of parameters on config.py
# or each attack.py in methods folder

attackName='FGSM' # FGSM | PGD | CW | DeepFool
dataset='cifar10' # cifar10 | cifar100
img_size=32
batch_size=64
target_classifier='resnet18' # resnet18 | resnet50 | resnet101
pretrained_dir='./resnet/pretrained_models/'
gpu_ids=(0 1)

# Change below values if you want
## PGD
pgd_iters=40
pgd_eps=0.003
pgd_alpha="2/255"
pgd_random='false'

## CW
cw_c="1e-4"
cw_kappa=0.0
cw_iters=10000
cw_lr=0.01
cw_binary_search=9
cw_targeted='false'

## FGSM
fgsm_eps=0.003

## DeepFool
deepFool_iters=5

# Attack and save images as .pth
echo "Attack $attackName is started"
python main.py --attack-name $attackName --device-ids ${gpu_ids[@]}\
 --dataset $dataset --image-size $img_size --batch-size $batch_size\
 --classifier $target_classifier --pretrained-dir $pretrained_dir\
 --pgd-iters $pgd_iters --pgd-eps $pgd_eps --pgd-alpha $pgd_alpha\
 --pgd-random-start $pgd_random\
 --cw-c $cw_c --cw-kappa $cw_kappa --cw-iters $cw_iters --cw-lr $cw_lr\
 --cw-binary-search-steps $cw_binary_search --cw-targeted $cw_targeted\
 --fgsm-eps $fgsm_eps\
 --deepfool-iters $deepFool_iters
