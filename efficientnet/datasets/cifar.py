from torch.utils import data
from torchvision import datasets, transforms


class CIFAR10DataLoader(data.DataLoader):

    def __init__(self, root: str, image_size: int, train: bool, batch_size: int, shuffle: bool, **kwargs):
        if train:
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(25),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])

        dataset = datasets.CIFAR10(root, train=train, transform=transform, download=True)

        super(CIFAR10DataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def cifar10_dataloaders(root='data', image_size=32, batch_size=128, **kwargs):
    train_loader = CIFAR10DataLoader(root, image_size, train=True, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = CIFAR10DataLoader(root, image_size, train=False, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader
