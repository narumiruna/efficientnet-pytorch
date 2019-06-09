from .utils import AttrDict, load_yaml


class Config(AttrDict):

    distributed = None

    @classmethod
    def from_yaml(cls, f):
        data = load_yaml(f)
        return cls(**data)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('

        for k, v in self.items():
            format_string += f'\n\t{k}: {v}'

        format_string += '\n)'

        return format_string
