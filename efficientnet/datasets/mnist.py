from torch.utils import data
from torchvision import datasets, transforms


class MNISTDataLoader(data.DataLoader):

    def __init__(self, root: str, image_size: int, train: bool, batch_size: int, shuffle: bool, **kwargs):
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        dataset = datasets.MNIST(
            root,
            train=train,
            transform=transform,
            download=True,
        )

        super(MNISTDataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def mnist_dataloaders(root='data', image_size=32, batch_size=128, **kwargs):
    train_loader = MNISTDataLoader(root, image_size, train=True, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = MNISTDataLoader(root, image_size, train=False, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader
