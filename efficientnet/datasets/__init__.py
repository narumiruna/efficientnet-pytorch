import sys

from ..utils import get_factory
from .mnist import mnist_dataloaders

DatasetFactory = get_factory(sys.modules[__name__])
