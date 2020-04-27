"""
DeepFool

This code is written by Seugnwan Seo
"""
import torch
from attacks import Attack

class DeepFool(Attack):
    """Reproduce DeepFool
    in the paper 'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'

    Arguments:
        target_cls {nn.Module} -- Target classifier to fool
        n_iters {n_iters} -- Step size
    """
    def __init__(self, target_cls, n_iters=5):
        super(DeepFool, self).__init__("DeepFool", target_cls)
        self.n_iters = n_iters

    def forward(self, imgs, _):
        imgs = imgs.to(self.device)

        for idx, img in enumerate(imgs):
            img.requires_grad = True

            output = self.target_cls(img)[0]

            _, first_predict = torch.max(output, 0)
            first_max = output[first_predict]
            grad_first = torch.autograd.grad(first_max, img)[0]

            num_classes = len(output)

            for _ in range(self.n_iters):
                img.requires_grad = True
                output = self.target_cls(img)[0]
                _, predict = torch.max(output, 0)

                if predict != first_predict:
                    img = torch.clamp(img, min=0, max=1).detach()
                    break

                r = None
                min_value = None

                for k in range(num_classes):
                    if k == first_predict:
                        continue

                    k_max = output[k]
                    grad_k = torch.autograd.grad(k_max, img)[0]

                    prime_max = k_max - first_max
                    grad_prime = grad_k - grad_first
                    value = torch.abs(prime_max)/torch.norm(grad_prime)

                    if r is None:
                        r = (torch.abs(prime_max)/(torch.norm(grad_prime)**2))*grad_prime
                        min_value = value
                    else:
                        if min_value > value:
                            r = (torch.abs(prime_max)/(torch.norm(grad_prime)**2))*grad_prime
                            min_value = value

                img = torch.clamp(img+r, min=0, max=1).detach()

            imgs[idx:idx+1, :, :, :] = img

        return imgs
