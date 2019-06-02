from torch.utils import data
from torchvision import datasets, transforms


class Expand(object):

    def __init__(self, dim=3):
        self.dim = dim

    def __call__(self, img):
        _, h, w = img.size()
        return img.expand(self.dim, h, w)


class MNISTDataloader(data.DataLoader):

    def __init__(self, root: str, image_size: int, train: bool, batch_size: int, shuffle: bool):
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            Expand(),
        ])

        dataset = datasets.MNIST(
            root,
            train=train,
            transform=transform,
            download=True,
        )

        super(MNISTDataloader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


def mnist_dataloaders(root='data', image_size=32, batch_size=128):
    train_loader = MNISTDataloader(root, image_size, train=True, batch_size=batch_size, shuffle=True)
    test_loader = MNISTDataloader(root, image_size, train=False, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
