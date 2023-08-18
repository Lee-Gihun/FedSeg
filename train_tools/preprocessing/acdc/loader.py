import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.utils.data as data

from .datasets import ACDC


def _data_transforms_acdc():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = A.Compose(
        [
            A.Normalize(mean=mean, std=std),
            A.RandomCrop(512, 512, p=1.0),
            # A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2(),
        ]
    )

    valid_transform = A.Compose(
        [A.Normalize(mean=mean, std=std), A.CenterCrop(512, 512, p=1.0), ToTensorV2(),]
    )

    return train_transform, valid_transform


def get_dataloader_acdc(
    root, train=True, batch_size=32, client_id=None, out_client=False
):
    train_transform, valid_transform = _data_transforms_acdc()

    transform = train_transform if train else valid_transform
    dataset = ACDC(root, train, transform, client_id, out_client)
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    return dataloader
