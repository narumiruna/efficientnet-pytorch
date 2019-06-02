import torch

from efficientnet.models import EfficientNet


def numel(model):
    return sum(p.numel() for p in model.parameters())


def test_output(model, size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()

    x = torch.randn(1, 3, size, size).to(device)
    with torch.no_grad():
        y = model(x)
        assert y.size(1) == 1000


def print_num_parameters(model):
    num_params = numel(model)
    print(f'Number of parameters: {num_params}')


def main():
    params = {
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }

    for arch, (w, d, s, r) in params.items():
        model = EfficientNet(w, d, r)

        print(f'Arch: {arch}, settings: {(w, d, s, r)}')
        test_output(model, s)
        print_num_parameters(model)


if __name__ == "__main__":
    main()
