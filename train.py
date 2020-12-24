import argparse
import os

import mlconfig
import torch
from torch import distributed
from torch import nn

from efficientnet.utils import distributed_is_initialized


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/mnist.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument('--no_cuda', action='store_true')
    return parser.parse_args()


def main():
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    config = mlconfig.load(args.config)
    print(config)

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device(f'cuda:{args.local_rank}' if use_cuda else 'cpu')

    if 'WORLD_SIZE' in os.environ:
        distributed.init_process_group(backend=args.backend)

    model = config.model()
    model.to(device)

    if use_cuda and distributed_is_initialized():
        torch.cuda.set_device(args.local_rank)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    optimizer = config.optimizer(model.parameters())
    scheduler = config.scheduler(optimizer)
    train_loader = config.dataset(train=True)
    valid_loader = config.dataset(train=False)

    trainer = config.trainer(model, optimizer, train_loader, valid_loader, scheduler, device)

    if args.resume is not None:
        trainer.resume(args.resume)

    trainer.fit()


if __name__ == "__main__":
    main()
