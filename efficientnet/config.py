from .utils import AttrDict, load_yaml


class Config(AttrDict):
    BATCH_SIZE: int = 2048

    @classmethod
    def from_yaml(cls, f):
        data = load_yaml(f)
        return cls(**data)
