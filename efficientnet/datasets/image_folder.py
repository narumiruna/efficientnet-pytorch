import os
from typing import List

import mlconfig
from PIL import Image
from torch.utils import data
from torchvision import datasets
from torchvision import transforms

from ..utils import distributed_is_initialized

@mlconfig.register
class ImageFolderDataLoader(data.DataLoader):

    def __init__(self,
                 root: str,
                 image_size: int,
                 train: bool,
                 batch_size: int,
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225],
                 **kwargs):
        normalize = transforms.Normalize(mean=mean, std=std)

        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(image_size + 32, interpolation=Image.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])

        phase = 'train' if train else 'val'
        dataset = datasets.ImageFolder(os.path.join(root, phase), transform=transform)

        sampler = None
        if train and distributed_is_initialized():
            sampler = data.distributed.DistributedSampler(dataset)

        super(ImageFolderDataLoader, self).__init__(dataset,
                                                    batch_size=batch_size,
                                                    shuffle=(sampler is None),
                                                    sampler=sampler,
                                                    **kwargs)
