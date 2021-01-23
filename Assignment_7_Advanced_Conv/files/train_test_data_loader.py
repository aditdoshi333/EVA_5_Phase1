import torch
from torchvision import datasets, transforms


def train_test_data_loader(train_transforms, test_transforms):

    # Train and test dataset
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
    testset =  datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)

    # Checking GPU is available or not
    SEED = 1
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)

    # Data loader args that will be passed to dataloader function
    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True,
                                                                                                           batch_size=64)

    # train dataloader
    train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)

    return train_loader, test_loader

