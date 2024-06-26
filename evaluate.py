import argparse

import torch
import torch.nn.functional as F
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

from efficientnet import models
from efficientnet.datasets.imagenet import ImageNetDataLoader
from efficientnet.models.efficientnet import params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="efficientnet_b0")
    parser.add_argument("-r", "--root", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("-w", "--weight", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, valid_loader, device):
    model.eval()

    loss_metric = MeanMetric()
    acc_metric = MulticlassAccuracy()

    valid_loader = tqdm(valid_loader, desc="Validate", ncols=0)
    for x, y in valid_loader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = F.cross_entropy(output, y)

        loss_metric.update(loss.item(), number=x.size(0))
        acc_metric.update(output.cpu(), y.cpu())

        eval_loss = float(loss_metric.compute())
        eval_acc = float(acc_metric.compute())
        valid_loader.set_postfix_str(f"valid loss: {eval_loss:.4f}, valid acc: {eval_acc:.4f}.")

    return loss_metric, acc_metric


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    model = getattr(models, args.arch)(pretrained=(args.weight is None))
    if args.weight is not None:
        state_dict = torch.load(args.weight, map_location="cpu")
        model.load_state_dict(state_dict)
    model.to(device)

    image_size = params[args.arch][2]
    valid_loader = ImageNetDataLoader(args.root, image_size, False, args.batch_size, num_workers=args.num_workers)

    evaluate(model, valid_loader, device)
