from torchvision import transforms


def train_test_transform(normalize=False, mean=None, std=None):
    train_transform_list = []
    test_transform_list = []

    train_transform_list.append(transforms.ToTensor())
    test_transform_list.append(transforms.ToTensor())

    if normalize:
        train_transform_list.append(transforms.Normalize(mean, std))
        test_transform_list.append(transforms.Normalize(mean, std))

    train_transforms = transforms.Compose(train_transform_list)
    test_transforms = transforms.Compose(test_transform_list)

    return train_transforms, test_transforms