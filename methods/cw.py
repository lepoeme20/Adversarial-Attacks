"""[summary]
"""
import torch
from attacks import Attack

class CW(Attack):
    def __init__(self, target_cls, args):
        super(CW, self).__init__("CW", target_cls)
        self.targeted = args.targeted
        self.c = args.cw_c
        self.kappa = args.cw_kappa
        self.n_iters = args.cw_iters
        self.lr = args.cw_lr
        self.binary_search_steps = args.cw_binary_search_steps

    def forward(self, imgs, labels):
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)

        imgs = imgs.detach().clone()
        x_arctanh = self.arctanh(imgs)

        for _ in range(self.binary_search_steps):
            delta = torch.zeros_like(imgs).to(self.device)
            optimizer = torch.optim.Adam(delta.parameters(), lr=self.lr)
            prev_loss = 1e6

            for step in range(self.n_iters):
                optimizer.zero_grad()
                adv_examples = self.scaler(x_arctanh + delta)
                loss1 = torch.sum(self.c*self._f(adv_examples, labels))
                loss2 = torch.functional.F.mse_loss(adv_examples, imgs, reduction='sum')

                loss = loss1 + loss2
                loss.backward()
                optimizer.step()

                if step % (self.n_iters // 10) == 0:
                    if loss > prev_loss:
                        break

                    prev_loss = loss

            adv_imgs = self.scaler(x_arctanh + delta).detach()
            return adv_imgs

    def _f(self, adv_imgs, labels):
        outputs = self.target_cls(adv_imgs)
        y_onehot = torch.nn.functional.one_hot(labels)

        real = (y_onehot * outputs).sum(dim=1)
        other, _ = torch.max((1-y_onehot)*outputs, dim=1)

        if self.targeted:
            loss = torch.clamp(other-real, min=-self.kappa)
        else:
            loss = torch.clamp(real-other, min=-self.kappa)

        return loss

    def arctanh(self, imgs):
        scaling = torch.clamp(imgs, max=1, min=-1)
        x = 0.999999 * scaling

        return 0.5*torch.log((1+x)/(1-x))

    def scaler(self, x_atanh):
        return ((torch.tanh(x_atanh))+1) * 0.5
