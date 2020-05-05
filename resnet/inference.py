"""Test your trained model
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from resnet import config
from resnet.classifier import classifier


def inference(args):
    """Main function to inference your trained model
    """
    if args.dataset.lower() == 'cifar10':
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

        testset = torchvision.datasets.CIFAR10(
            root='datasets/cifar10', download=True, train=False, transform=transform)
        _, testset = torch.utils.data.random_split(
            testset, [int(len(testset) * .2), int(len(testset) * .8)])
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

    elif args.dataset.lower() == 'cifar100':
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                ),
            ])

        testset = torchvision.datasets.CIFAR100(
            root='datasets/cifar100', download=True, train=False, transform=transform
            )
        _, testset = torch.utils.data.random_split(
            testset, [int(len(testset) * .2), int(len(testset) * .8)]
            )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers
            )

    net = classifier(
        dataset=args.dataset, clf=args.classifier, train=args.train,
        pretrained_dir=args.pretrained_dir)

    net.to(args.device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    net.eval()
    correct = 0
    with torch.no_grad():
        for data in testloader:
            imgs, labels = data
            imgs, labels = imgs.to(args.device), labels.to(args.device)

            outputs = net(imgs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    print(
        'Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / len(testset)))

if __name__ == "__main__":
    opt = config.get_config()
    print(opt)

    inference(opt)
