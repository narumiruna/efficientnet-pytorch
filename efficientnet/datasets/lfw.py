import glob
import os
import random

import torch
import torchvision.transforms.functional as F
from torch.utils import data
from torchvision import transforms
from torchvision.datasets.folder import pil_loader


class JointRandomFlip(object):

    def __call__(self, images):
        if random.random() < 0.5:
            images = [F.vflip(img) for img in images]

        if random.random() < 0.5:
            images = [F.hflip(img) for img in images]

        return images


class JointRandomRotate(object):

    def __call__(self, images):
        angle = random.randint(0, 360)
        return [img.rotate(angle) for img in images]


class JointRandomResizeCrop(object):

    def __init__(self, size, scale=(0.08, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, images):
        scale = random.uniform(*self.scale)

        w, h = images[0].size

        cw, ch = int(w * scale), int(h * scale)

        i = random.randint(0, w - cw)
        j = random.randint(0, h - ch)
        box = (i, j, i + cw, j + ch)

        return [F.resize(img.crop(box), self.size) for img in images]


class LFW(data.Dataset):
    img_dir = 'lfw_funneled'
    mask_dir = 'parts_lfw_funneled_gt_images'

    def __init__(self, root, transform=None, joint_transform=None):
        self.root = root
        self.transform = transform
        self.joint_transform = joint_transform

        self.img_paths = glob.glob(os.path.join(root, self.img_dir, '*/*.jpg'))
        self.mask_paths = glob.glob(os.path.join(root, self.mask_dir, '*.ppm'))

    def __getitem__(self, index):
        mask_path = self.mask_paths[index]
        basename = os.path.basename(mask_path)
        name = '_'.join(basename.split('_')[:-1])

        img_path = os.path.join(self.root, self.img_dir, name, basename.replace('.ppm', '.jpg'))

        img = pil_loader(img_path)
        mask = pil_loader(mask_path)

        if self.joint_transform:
            img, mask = self.joint_transform([img, mask])

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        # (hair, non - hair)
        mask = torch.cat([mask[0:1], 1 - mask[0:1]], dim=0)

        return img, mask

    def __len__(self):
        return len(self.mask_paths)


def lfw_dataloaders(root, batch_size, image_size=24, valid_ratio=0.1, workers=0):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4332, 0.3757, 0.3340], std=[0.2983, 0.2732, 0.2665]),
    ])

    joint_transform = transforms.Compose([
        JointRandomFlip(),
        JointRandomRotate(),
        JointRandomResizeCrop(image_size),
    ])

    assert 0.0 <= valid_ratio < 1.0

    dataset = LFW(root, transform=transform, joint_transform=joint_transform)

    if valid_ratio == 0.0:
        train_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        return train_loader
    else:
        size = len(dataset)
        val_size = int(size * valid_ratio)

        train_dataset, val_dataset = data.random_split(dataset, [size - val_size, val_size])

        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

        val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

        return train_loader, val_loader
