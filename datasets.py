import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import os



def load_tinyimagenet(root_dir="./data", batch_size=64, flatten=True, samples_dist=0):
    mean = torch.tensor([0.4802, 0.4481, 0.3975])
    std = torch.tensor([0.2302, 0.2265, 0.2262])
    normalize = transforms.Normalize(mean=mean, std=std)
    train_set = datasets.ImageFolder(os.path.join(root_dir,"tiny-imagenet-200","train"),
                                          transform=transforms.Compose([
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(56, padding_mode='edge'),
                                            #   normalize,
                                              transforms.ToTensor(),
                                          ]))
    test_set = datasets.ImageFolder(os.path.join(root_dir,"tiny-imagenet-200","val"),
                                         transform=transforms.Compose([
                                             # transforms.RandomResizedCrop(64, scale=(0.875, 0.875), ratio=(1., 1.)),
                                             transforms.CenterCrop(56),
                                             transforms.ToTensor(),
                                            #  normalize,
                                         ]))
    
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
    print("==>>> Trainig set size = {}".format(len(train_loader.dataset)))
    print("==>>> Test set size = {}".format(len(test_loader.dataset)))

    for loader in [train_loader, test_loader]:
        loader.mean, loader.std = mean, std
    return train_loader, test_loader

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

    print("==>>> Trainig set size = {}".format(len(train_loader.dataset)))
    print("==>>> Test set size = {}".format(len(test_loader.dataset)))

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

    print("==>>> Trainig set size = {}".format(len(train_loader.dataset)))
    print("==>>> Test set size = {}".format(len(test_loader.dataset)))

    return train_loader, test_loader
