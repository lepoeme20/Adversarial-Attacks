"""
FGSM

This code is written by Seugnwan Seo
"""
import torch
import torch.nn as nn
from attacks import Attack

class FGSM(Attack):
    """Reproduce Fast Gradients Sign Method (FGSM)
    in the paper 'Explaining and harnessing adversarial examples'

    Arguments:
        target_cls {nn.Module} -- Target classifier to fool
        eps {float} -- Magnitude of perturbation
    """
    def __init__(self, target_cls, eps=0.003):
        super(FGSM, self).__init__("FGSM", target_cls)
        self.eps = eps
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, imgs, labels):
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)

        imgs.requires_grad = True

        outputs = self.target_cls(imgs)
        loss = self.criterion(outputs, labels)

        gradients = torch.autograd.grad(loss, imgs)[0]

        adversarial_examples = imgs+(self.eps*gradients.sign())
        adversarial_examples = torch.clamp(adversarial_examples, min=0, max=1).detach()

        return adversarial_examples
