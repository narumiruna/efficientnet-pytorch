import os
from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as F
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

    def __init__(self, model, optimizer, train_loader, valid_loader, scheduler, device, num_epochs: int,
                 output_dir: str):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        self.device = device

        self.epoch = 1
        self.start_epoch = 1
        self.best_acc = 0

        self.checkpoint_path = os.path.join(self.output_dir, 'checkpoint.pth')

    def fit(self):
        os.makedirs(self.output_dir, exist_ok=True)

        if os.path.exists(self.checkpoint_path):
            self.restore_checkpoint()

        epoch_range = trange(self.start_epoch, self.num_epochs + 1, desc='Epoch', ncols=0)
        for self.epoch in epoch_range:
            self.scheduler.step()

            train_loss, train_acc = self.train()
            valid_loss, valid_acc = self.evaluate()

            if valid_acc.accuracy > self.best_acc:
                self.best_acc = valid_acc.accuracy
                self.save_checkpoint(self.epoch)

            epoch_range.set_postfix_str(f'Epoch: {self.epoch}/{self.num_epochs}, '
                                        f'train loss: {train_loss}, train acc: {train_acc}, '
                                        f'valid loss: {valid_loss}, valid acc: {valid_acc}, '
                                        f'best valid acc: {self.best_acc * 100:.2f}')

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

            pred = output.argmax(dim=1)

            train_loss.update(loss.item(), number=x.size(0))
            train_acc.update(pred, y)

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

                pred = output.argmax(dim=1)

                valid_loss.update(loss.item(), number=x.size(0))
                valid_acc.update(pred, y)

                valid_loader.set_postfix_str(f'valid loss: {valid_loss}, valid acc: {valid_acc}.')

        return valid_loss, valid_acc

    def save_checkpoint(self, epoch):
        self.model.eval()

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch,
            'best_acc': self.best_acc
        }

        torch.save(checkpoint, self.checkpoint_path)

    def restore_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.best_acc = checkpoint['best_acc']
