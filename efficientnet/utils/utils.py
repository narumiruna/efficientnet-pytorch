import yaml
from torch import distributed


def load_yaml(f):
    with open(f, 'r') as fp:
        return yaml.load(fp)


def distributed_is_initialized():
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False
