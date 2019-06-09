import argparse

import torch
from torch import distributed, nn

from efficientnet import Config
from efficientnet.datasets import DatasetFactory
from efficientnet.models import ModelFactory
from efficientnet.optim import OptimFactory, SchedulerFactory
from efficientnet.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/mnist.yaml')
    parser.add_argument('-r', '--root', type=str, help='Path to dataset.')
    parser.add_argument('--resume', type=str, default=None)
    return parser.parse_args()


def init_process(backend, init_method, rank, world_size):
    distributed.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )


def load_config():
    args = parse_args()
    config = Config.from_yaml(args.config)

    if args.root:
        config.dataset.root = args.root

    config.update(vars(args))

    return config


def main():
    torch.backends.cudnn.benchmark = True

    config = load_config()
    print(config)

    if config.distributed:
        init_process(**config.distributed)

    device = torch.device('cuda' if torch.cuda.is_available() and config.use_cuda else 'cpu')

    model = ModelFactory.create(**config.model)
    if config.distributed:
        model.to(device)
        model = nn.parallel.DistributedDataParallel(model)
    else:
        if config.data_parallel:
            model = nn.DataParallel(model)
        model.to(device)

    optimizer = OptimFactory.create(model.parameters(), **config.optimizer)
    scheduler = SchedulerFactory.create(optimizer, **config.scheduler)

    train_loader, valid_loader = DatasetFactory.create(**config.dataset)

    trainer = Trainer(config, model, optimizer, train_loader, valid_loader, scheduler, device)

    if config.resume is not None:
        trainer.resume(config.resume)

    trainer.fit()


if __name__ == "__main__":
    main()
