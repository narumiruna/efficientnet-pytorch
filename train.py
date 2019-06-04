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

    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--backend', type=str, default='gloo', help='Name of the backend to use.')
    parser.add_argument('--init-method', type=str, default='tcp://127.0.0.1:23456', help='URL specifying how to initialize the package.')
    parser.add_argument('--rank', type=int, default=0, help='Rank of the current process.')
    parser.add_argument('--world-size', type=int, default=1, help='Number of processes participating in the job.')

    return parser.parse_args()

def init_process(backend, init_method, rank, world_size):
    distributed.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size)


def load_config(args):
    config = Config.from_yaml(args.config)

    if args.root:
        config.dataset.root = args.root

    return config

def main():
    args = parse_args()
    config = load_config(args)
    print(config)

    if args.distributed:
        init_process(args.backend, args.init_method, args.rank, args.world_size)

    device = torch.device('cuda' if torch.cuda.is_available() and config.use_cuda else 'cpu')

    model = ModelFactory.create(**config.model)
    if distributed.is_initialized():
        model = nn.parallel.DistributedDataParallel(model)
    else:
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
