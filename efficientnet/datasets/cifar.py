import mlconfig
from torch.utils import data
from torchvision import datasets, transforms


@mlconfig.register
class CIFAR10DataLoader(data.DataLoader):

    def __init__(self, root: str, image_size: int, train: bool, batch_size: int, shuffle: bool = True, **kwargs):
        normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))

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

        dataset = datasets.CIFAR10(root, train=train, transform=transform, download=True)

        super(CIFAR10DataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
