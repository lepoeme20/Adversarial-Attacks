"""Train module for classifier
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from resnet.classifier import classifier


def train(args):
    """[summary]
    
    Arguments:
        args {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    save_path = os.path.join(args.pretrained_dir, args.dataset, args.classifier + '.pth')

    if args.dataset.lower() == 'cifar10':
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        )

        transform_dev = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010]
                ),
            ]
        )

        trainset = torchvision.datasets.CIFAR10(root='datasets/cifar10', download=True, train=True,
                                                transform=transform)
        devset = torchvision.datasets.CIFAR10(
            root='datasets/cifar10', download=True, train=False, transform=transform_dev)
        devset, _ = torch.utils.data.random_split(
            devset, [int(len(devset) * .2), int(len(devset) * .8)])

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
        devloader = torch.utils.data.DataLoader(
            devset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

    elif args.dataset.lower() == 'cifar100':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
                ),
            ])

        transform_dev = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
                ),
            ])

        trainset = torchvision.datasets.CIFAR100(
            root='datasets/cifar100', download=True, train=True, transform=transform)
        devset = torchvision.datasets.CIFAR100(
            root='datasets/cifar100', download=True, train=False, transform=transform_dev)
        devset, _ = torch.utils.data.random_split(
            devset, [int(len(devset) * .2), int(len(devset) * .8)])

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
        devloader = torch.utils.data.DataLoader(
            devset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

    net = classifier(
        dataset=args.dataset, clf=args.classifier, train=args.train,
        pretrained_dir=args.pretrained_dir)

    # layer = 0
    # for child in net.children():
    #     layer += 1
    #     if layer < 6:
    #         for param in child.parameters():
    #             param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr_cls, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=args.lr_cls, betas=(0.9, 0.999))
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    if torch.cuda.device_count() > 0:
        print('\n===> Training on GPU!')
        net = nn.DataParallel(net)

    best_acc = 0
    for epoch in range(args.epochs):
        print('\n===> epoch %d' % epoch)
        running_loss = 0.0
        test_loss = 0
        correct = 0
        total = 0

        for i, data in enumerate(trainloader, 0):
            net.train()

            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 20 == 19:  # print every 100 mini-batches
                print('{}/{} loss:{:.3f}'.format(i + 1, len(trainloader) + 1, running_loss / 20))
                running_loss = 0.0

        for idx, dev in enumerate(devloader):
            imgs, labels = dev
            imgs, labels = imgs.to(device), labels.to(device)

            net.eval()
            with torch.no_grad():
                outputs = net(imgs)

                # Loss
                loss = criterion(outputs, labels)
                test_loss += loss

                # Accuracy
                _, predicted = torch.max(outputs, 1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

                print('[Dev] {}/{} Loss: {:.3f}, Acc: {:.3f}'.format(
                    idx+1, len(devloader), test_loss/(idx+1), 100.*(correct/total)))


        # Save checkpoint
        acc = 100. * (correct/total)
        scheduler.step(test_loss)

        if acc > best_acc:
            print("Acc: {:.4f}".format(acc))
            best_acc = acc
            if torch.cuda.device_count() > 0:
                torch.save({
                    'model_state_dict': net.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, save_path)

    return net
