import torch
from torch import nn


def update_fn(v, value, decay):
    v -= (1 - decay) * (v - value)
    return v


class ExponentialMovingAverage(object):
    """https://github.com/tensorflow/tensorflow/blob/93dd14dce2e8751bcaab0a0eb363d55eb0cc5813/tensorflow/python/training/moving_averages.py#L252"""

    def __init__(self, decay=0.9999):
        self._decay = decay
        self._shadow = None

    def _register(self, model: nn.Module):
        self._shadow = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self._shadow[name] = torch.tensor(param, requires_grad=False)

    def update(self, model: nn.Module, num_updates=None):
        if self._shadow is None:
            self._register(model)

        if num_updates is not None:
            self._decay = min(self._decay, (1.0 + num_updates) / (10.0 + num_updates))

        state_dict = model.state_dict()
        for name, param in self._shadow.items():
            value = state_dict[name].clone()
            param = update_fn(param, value, self._decay)


def main():
    import torch
    from torch import optim
    from torchvision import models
    model = nn.Linear(1, 1, bias=False)

    ema = ExponentialMovingAverage(0.5)

    optimizer = optim.Adam(model.parameters(), lr=0.1)
    for _ in range(10):
        x = torch.randn(1, 1)
        y = model(x)

        loss = (x - y - 100).norm()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(ema._shadow, model.weight.data)
        ema.update(model)
        print(ema._shadow, model.weight.data)
        print('-' * 100)


if __name__ == '__main__':
    main()
