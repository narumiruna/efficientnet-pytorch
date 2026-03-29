import yaml
from torch import distributed


def load_yaml(f: str) -> None:
    with open(f) as fp:
        return yaml.load(fp)


def distributed_is_initialized() -> bool:
    if distributed.is_available():
        if distributed.is_initialized():
            return True
    return False
