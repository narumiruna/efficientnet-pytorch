import sys

from ..utils import get_factory
from .cifar import cifar10_dataloaders
from .custom import custom_dataloaders
from .imagenet import imagenet_dataloaders
from .mnist import mnist_dataloaders

DatasetFactory = get_factory(sys.modules[__name__])
