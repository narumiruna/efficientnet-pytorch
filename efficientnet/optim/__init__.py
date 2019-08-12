import sys

from torch import optim
from torch.optim import SGD

from ..utils import get_factory
from .rmsprop import TFRMSprop

OptimFactory = get_factory(sys.modules[__name__])
SchedulerFactory = get_factory(optim.lr_scheduler)
