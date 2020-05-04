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
gpu_ids=(0 1)

# Attack and save images as .pth
echo "Attack $FGSM is stared"
python main.py --attack-name $attackName --device-ids ${gpu_ids[@]}
