import torch

from efficientnet.models.efficientnet import EfficientNet, params


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
    for arch, (w, d, s, r) in params.items():
        model = EfficientNet(w, d, r)

        print(f'Arch: {arch}, settings: {(w, d, s, r)}')
        test_output(model, s)
        print_num_parameters(model)


if __name__ == "__main__":
    main()
