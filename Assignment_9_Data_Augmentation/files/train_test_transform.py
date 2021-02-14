import albumentations as A
from files.Albumentationtransform import AlbumentationTransforms


def train_test_transform(normalize=False, mean=None, std=None):
    train_transform_list = []
    test_transform_list = []

    train_transform_list.append(A.HorizontalFlip())
    train_transform_list.append(A.Rotate((-60.0, 60.0)))
    train_transform_list.append(A.CoarseDropout(max_holes=2, min_holes=1, max_height=8, max_width= 8, p=1))


    if normalize:
        train_transform_list.append(A.Normalize(mean= mean, std = std))
        test_transform_list.append(A.Normalize(mean= mean,std= std))

    train_transforms = AlbumentationTransforms(train_transform_list)
    test_transforms = AlbumentationTransforms(test_transform_list)

    return train_transforms, test_transforms