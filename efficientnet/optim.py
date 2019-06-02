from torch import optim

from .utils import get_factory

OptimFactory = get_factory(optim)
SchedulerFactory = get_factory(optim.lr_scheduler)
