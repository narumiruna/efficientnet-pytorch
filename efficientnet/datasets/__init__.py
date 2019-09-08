import sys

from ..utils import get_factory
from .cifar import CIFAR10DataLoader
from .imagenet import ImageNetDataLoader
from .mnist import MNISTDataLoader

DatasetFactory = get_factory(sys.modules[__name__])
