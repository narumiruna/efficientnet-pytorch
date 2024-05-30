import os

import mlconfig
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils import data
from torchmetrics import MeanMetric
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm
from tqdm import trange

from .models import EfficientNet


@mlconfig.register
class Trainer:
    def __init__(
        self,
        model: EfficientNet,
        optimizer: optim.Optimizer,
        train_loader: data.DataLoader,
        valid_loader: data.DataLoader,
        scheduler: optim.lr_scheduler.LRScheduler,
        device: torch.device,
        num_epochs: int,
        output_dir: str,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = output_dir

        self.num_classes = self.model.num_classes

        self.epoch = 1
        self.best_acc = 0
        self.metrics = MetricCollection(
            {
                "train_loss": MeanMetric(),
                "train_acc": MulticlassAccuracy(num_classes=self.num_classes),
                "valid_loss": MeanMetric(),
                "valid_acc": MulticlassAccuracy(num_classes=self.num_classes),
            }
        )

    def format_metrics(self) -> str:
        return

    def fit(self) -> None:
        epochs = trange(self.epoch, self.num_epochs + 1, desc="Epoch")
        for self.epoch in epochs:
            self.train()
            self.validate()
            self.scheduler.step()

            self.save_checkpoint(os.path.join(self.output_dir, "checkpoint.pth"))
            valid_acc = float(self.metrics["valid_acc"].compute())
            if valid_acc > self.best_acc:
                self.best_acc = valid_acc
                self.save_checkpoint(os.path.join(self.output_dir, "best.pth"))

            format_string = f"Epoch: {self.epoch}/{self.num_epochs}"
            for k, v in self.metrics.items():
                format_string += f", {k}: {v.compute():.4f}"
            format_string += f", best acc: {self.best_acc:.4f}\n"
            tqdm.write(format_string)

    def train(self) -> None:
        self.model.train()

        train_loader = tqdm(self.train_loader, desc="Train")
        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x)
            loss = F.cross_entropy(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.metrics["train_loss"].update(loss.item(), weight=x.size(0))
            self.metrics["train_acc"].update(output.cpu(), y.cpu())

    @torch.no_grad()
    def validate(self) -> None:
        self.model.eval()

        valid_loader = tqdm(self.valid_loader, desc="Validate")
        for x, y in valid_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x)
            loss = F.cross_entropy(output, y)

            self.metrics["valid_loss"].update(loss.item(), weight=x.size(0))
            self.metrics["valid_acc"].update(output.cpu(), y.cpu())

    def save_checkpoint(self, f: str) -> None:
        self.model.eval()

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "best_acc": self.best_acc,
        }

        dirname = os.path.dirname(f)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        torch.save(checkpoint, f)

    def resume(self, f: str) -> None:
        checkpoint = torch.load(f, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

        self.epoch = checkpoint["epoch"] + 1
        self.best_acc = checkpoint["best_acc"]
