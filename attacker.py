import os
import torch
import torchvision
import torchvision.transforms as transforms
from utils import network_initialization
import attacks
from methods import pgd, deepfool, fgsm, pgd

def attack(args):
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

    target_cls = network_initialization(args)
    attack_module = globals()[args.attack_name.lower()]
    attack_func = getattr(attack_module, args.attack_name)
    attack = attack_func(target_cls, args)
    save_path = os.path.join("Adv_examples", args.dataset.lower())
    attack.inference(data_loader=dataloader, save_path=save_path, file_name=args.attack_name+".pt")
