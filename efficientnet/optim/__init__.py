from mlconfig.torch import register_torch_optimizers
from mlconfig.torch import register_torch_schedulers

from .rmsprop import TFRMSprop

register_torch_optimizers()
register_torch_schedulers()
