"""[summary]

"""
import torch
import torch.nn as nn
from attacks import Attack

class PGD(Attack):
    """Reproduce PGD
    in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'

    Arguments:
        target_cls {nn.Module} -- Target classifier to fool
        eps {float} -- For bound eta (difference between original imgs and adversarial imgs)
        alpha {float} -- Magnitude of perturbation (same as eps in FGSM)
        n_iters {int} -- Step size
        random_start {bool} -- If ture, initialize perturbation using eps
    """
    def __init__(self, target_cls, args):
        super(PGD, self).__init__("PGD", target_cls)
        self.eps = args.pgd_eps
        self.alpha = args.pgd_alpha
        self.n_iters = args.pgd_iters
        self.random_start = args.pgd_random_start
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, imgs, labels):
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)

        clone_imgs = imgs.clone().detach()

        if self.random_start:
            imgs = imgs + torch.empty_like(imgs).uniform_(-self.eps, self.eps)
            imgs = torch.clamp(imgs, min=0, max=1)

        for _ in range(self.n_iters):
            imgs.requires_grad = True
            outputs = self.target_cls(imgs)
            loss = self.criterion(outputs, labels)

            grad = torch.autograd.grad(loss, imgs)[0]
            adversarial_examples = imgs + self.alpha*grad.sign()
            eta = torch.clamp(adversarial_examples - clone_imgs, min=-self.eps, max=self.eps)
            imgs = torch.clamp(clone_imgs + eta, min=0, max=1).detach()

        return imgs, labels
