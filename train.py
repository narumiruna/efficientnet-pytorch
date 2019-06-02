import argparse

import torch
from torch import nn

from efficientnet import Config
from efficientnet.datasets import DatasetFactory
from efficientnet.models import ModelFactory
from efficientnet.optim import OptimFactory, SchedulerFactory
from efficientnet.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/mnist.yaml')
    parser.add_argument('-r', '--root', type=str, help='Path to dataset.')
    return parser.parse_args()


def load_config():
    args = parse_args()
    config = Config.from_yaml(args.config)

    if args.root:
        config.dataset.root = args.root
    return config


def main():
    config = load_config()
    print(config)

    device = torch.device('cuda' if torch.cuda.is_available() and config.use_cuda else 'cpu')

    model = ModelFactory.create(**config.model)
    if config.data_parallel:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = OptimFactory.create(model.parameters(), **config.optimizer)
    scheduler = SchedulerFactory.create(optimizer, **config.scheduler)

    train_loader, valid_loader = DatasetFactory.create(**config.dataset)

    trainer = Trainer(model, optimizer, train_loader, valid_loader, scheduler, device, config.num_epochs,
                      config.output_dir)
    trainer.fit()


if __name__ == "__main__":
    main()
