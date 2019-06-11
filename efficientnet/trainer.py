import os
import shutil
from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data
from tqdm import tqdm, trange

from .metrics import Accuracy, Average


class AbstractTrainer(metaclass=ABCMeta):

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError


class Trainer(AbstractTrainer):

    def __init__(
            self,
            model: nn.Module,
            optimizer: optim.Optimizer,
            train_loader: data.DataLoader,
            valid_loader: data.DataLoader,
            scheduler: optim.lr_scheduler._LRScheduler,
            device: torch.device,
            output_dir: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.output_dir = output_dir

        self.start_epoch = 1
        self.best_acc = 0

    def fit(self, num_epochs):
        epochs = trange(self.start_epoch, num_epochs + 1, desc='Epoch', ncols=0)
        for epoch in epochs:
            self.scheduler.step()

            train_loss, train_acc = self.train()
            valid_loss, valid_acc = self.evaluate()

            last_checkpoint = os.path.join(self.output_dir, 'checkpoint.pth')
            best_checkpoint = os.path.join(self.output_dir, 'best.pth')
            if valid_acc.accuracy > self.best_acc:
                self.best_acc = valid_acc.accuracy
                self.save_checkpoint(epoch, last_checkpoint)
                shutil.copy(last_checkpoint, best_checkpoint)
            else:
                self.save_checkpoint(epoch, last_checkpoint)

            epochs.set_postfix_str(f'Epoch: {epoch}/{num_epochs}, '
                                   f'train loss: {train_loss}, train acc: {train_acc}, '
                                   f'valid loss: {valid_loss}, valid acc: {valid_acc}, '
                                   f'best valid acc: {self.best_acc:.2f}')

    def train(self):
        self.model.train()

        train_loss = Average()
        train_acc = Accuracy()

        train_loader = tqdm(self.train_loader, ncols=0, desc='Train')
        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x)
            loss = F.cross_entropy(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(), number=x.size(0))
            train_acc.update(output, y)

            train_loader.set_postfix_str(f'train loss: {train_loss}, train acc: {train_acc}.')

        return train_loss, train_acc

    def evaluate(self):
        self.model.eval()

        valid_loss = Average()
        valid_acc = Accuracy()

        with torch.no_grad():
            valid_loader = tqdm(self.valid_loader, desc='Validate', ncols=0)
            for x, y in valid_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                loss = F.cross_entropy(output, y)

                valid_loss.update(loss.item(), number=x.size(0))
                valid_acc.update(output, y)

                valid_loader.set_postfix_str(f'valid loss: {valid_loss}, valid acc: {valid_acc}.')

        return valid_loss, valid_acc

    def save_checkpoint(self, epoch, f):
        self.model.eval()

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch,
            'best_acc': self.best_acc
        }

        dirname = os.path.dirname(f)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        torch.save(checkpoint, f)

    def resume(self, f):
        checkpoint = torch.load(f, map_location='cpu')

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']
