# Adversarial Attacks
This repository provide famous adversarial attacks.

<br>

## Dependencies
- python 3.6.1
- pytorch 1.4.0

## Papers
- Explaining and harnessing adversarial example: [FGSM](https://arxiv.org/pdf/1412.6572.pdf)
- Towards Evaluating the Robustness of Neural Networks: [CW](https://arxiv.org/pdf/1608.04644.pdf)
- Towards Deep Learning Models Resistant to Adversarial Attacks: [PGD](https://arxiv.org/pdf/1706.06083.pdf)
- DeepFool: a simple and accurate method to fool deep neural networks: [DeepFool](https://arxiv.org/pdf/1511.04599.pdf)

## Usage
# 
**Multi GPUs are allowed**
### Attacks
```bash
foo@bar:.../Attack-repo$ ./attack.sh
```

if you want to change hyper-parameter such as attack method or epsilons for attack quality, open the attack.sh file and just change arguments.

Open the file on terminal or your favorite editor,

```bash
foo@bar:.../Attack-repo$ vim attack.sh
```

and change values in "Set parameters" block.

You can check description in config.py

### Save images
```bash
foo@bar:.../Attack-repo$ ./visualize.sh
```
You can set option to save all images or just the one you have selected.

```bash
foo@bar:.../Attack-repo$ vim visualize.sh
```
- parameters
    - normal: Set 'true' when you need save normal images. If 'false', adversarial examples will be saved
    - n_rows: Number of rows in saved figure
    - batch_size: Mini batch-size for torch.utils.data.DataLoader. If you don't need to compare images one by one, you can use a size as large as your GPU resource allows.
    - set_idx: If you set 'true', only the image of index belonging to the indices variable.specified below will be saved. On the other hand, if set to 'false', all images will be saved.
    - indices: Image indices to save.
    - attack_method: 'DeepFool', 'FGSM', 'PGD', and 'DeepFool' are allowed. **You have to be careful about the case.**
    - dataset: 'cifar10' and 'cifar100' are allowed.
