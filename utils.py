import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from resnet.classifier import classifier


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def network_initialization(args):
    target_cls = Classifier(args)

    # Using multi GPUs if you have
    if torch.cuda.device_count() > 0:
        target_cls = nn.DataParallel(target_cls, device_ids=args.device_ids)

    # change device to set device (CPU or GPU)
    target_cls.to(args.device)

    for params in target_cls.parameters():
        params.requires_grad = False

    return target_cls

def get_dataloader(args):
    if args.dataset.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ), ])
        dataset = torchvision.datasets.CIFAR10(
            root='datasets/cifar10', download=True, train=False, transform=transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu)

    elif args.dataset.lower() == 'cifar100':
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            ), ])
        dataset = torchvision.datasets.CIFAR100(
            root='datasets/cifar100', download=True, train=False, transform=transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    return dataloader

class Classifier(nn.Module):
    '''
    Load target classifier
    '''
    def __init__(self, args):
        super(Classifier, self).__init__()

        self.net = classifier(dataset=args.dataset, clf=args.classifier,\
            train=False, pretrained_dir=args.pretrained_dir)

    def forward(self, x):
        out = self.net(x)
        return out
