import os
import warnings

from torch.utils import data
from torchvision import datasets, transforms

from ..utils import distributed_is_initialized


class ImageNetDataLoader(data.DataLoader):

    def __init__(self, root: str, image_size: int, train: bool, batch_size: int, **kwargs):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        warnings.filterwarnings('ignore', category=UserWarning)

        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(int(image_size * 256 / 224)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])

        split = 'train' if train else 'val'
        dataset = datasets.ImageFolder(os.path.join(root, split), transform=transform)

        sampler = None
        if train and distributed_is_initialized():
            sampler = data.distributed.DistributedSampler(dataset)

        super(ImageNetDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, **kwargs)


def imagenet_dataloaders(root: str, image_size: int, batch_size: int, **kwargs):
    train_loader = ImageNetDataLoader(root, image_size, True, batch_size, **kwargs)
    valid_loader = ImageNetDataLoader(root, image_size, False, batch_size, **kwargs)
    return train_loader, valid_loader
