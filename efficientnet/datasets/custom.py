import os
from glob import glob

from torch.utils import data
from torchvision import transforms
from torchvision.datasets.folder import pil_loader


class CustomDataset(data.Dataset):

    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        if self.train:
            image_dir = os.path.join(self.root, 'train')
        else:
            image_dir = os.path.join(self.root, 'valid')

        self.paths = glob(os.path.join(image_dir, '*.jpg'))

    def __getitem__(self, index):
        path = self.paths[index]
        img = pil_loader(path)
        target = 0 if 'cat' in path else 1

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.paths)


class CustomDataLoader(data.DataLoader):

    def __init__(self, root: str, image_size: int, batch_size: int, train: bool = True, **kwargs):
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset = CustomDataset(root, train=train, transform=transform)
        super(CustomDataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=True, **kwargs)


def custom_dataloaders(root='data', image_size=224, batch_size=128, **kwargs):
    train_loader = CustomDataLoader(root, image_size, batch_size=batch_size, train=True, **kwargs)
    test_loader = CustomDataLoader(root, image_size, batch_size=batch_size, train=False, **kwargs)
    return train_loader, test_loader
