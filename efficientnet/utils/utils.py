import yaml


def load_yaml(f):
    with open(f, 'r') as fp:
        return yaml.load(fp)
