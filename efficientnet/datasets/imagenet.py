from torch.utils import data
from torchvision import datasets, transforms


class ImageNetDataLoader(data.DataLoader):

    def __init__(self, root: str, image_size: int, train: bool, batch_size: int, download: bool = True, use_distributed=False, **kwargs):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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
        dataset = datasets.ImageNet(root, split, download=download, transform=transform)

        sampler = None
        if train and use_distributed:
            sampler = data.distributed.DistributedSampler(dataset)

        super(ImageNetDataLoader, self).__init__(
            dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, **kwargs)


def imagenet_dataloaders(root: str, image_size: int, batch_size: int, download: bool = True, **kwargs):
    train_loader = ImageNetDataLoader(root, image_size, True, batch_size, download=download, **kwargs)
    valid_loader = ImageNetDataLoader(root, image_size, False, batch_size, download=download, **kwargs)
    return train_loader, valid_loader
