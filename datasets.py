import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import os


def load_mnist(root_dir="./data", batch_size=512, flatten=True, samples_dist=0):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    if flatten:
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.0,), (1.0,)),
                transforms.Lambda(torch.flatten),
            ]
        )
    else:
        trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))]
        )
    train_set = datasets.MNIST(
        root=root_dir, train=True, transform=trans, download=True
    )
    test_set = datasets.MNIST(
        root=root_dir, train=False, transform=trans, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=8
    )
    if samples_dist > 0:
        sampler = torch.utils.data.SubsetRandomSampler(
            [i for i in range(0, 10000, samples_dist)]
        )
    else:
        sampler = None
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        sampler=sampler,
    )
    return train_loader, test_loader


def load_cifar10(root_dir="./data", batch_size=64, flatten=True, samples_dist=0):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    if flatten:
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.0,), (1.0,)),
                transforms.Lambda(torch.flatten),
            ]
        )
    else:
        trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))]
        )
    train_set = datasets.CIFAR10(
        root=root_dir, train=True, transform=trans, download=True
    )
    test_set = datasets.CIFAR10(
        root=root_dir, train=False, transform=trans, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    if samples_dist > 0:
        sampler = torch.utils.data.SubsetRandomSampler(
            [i for i in range(0, 10000, samples_dist)]
        )
    else:
        sampler = None
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        sampler=sampler,
    )

    return train_loader, test_loader
