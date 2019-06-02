import sys

from ..utils import get_factory
from .efficientnet import (efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4,
                           efficientnet_b5, efficientnet_b6, efficientnet_b7)

ModelFactory = get_factory(sys.modules[__name__])
