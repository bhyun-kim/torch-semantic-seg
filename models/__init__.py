from .losses import * 
from .heads import * 
from .lr_schedulers import * 

from registry import Registry

SchedulerRegistry = Registry()
WrapperRegistry = Registry()
OptimRegistry = Registry()

__all__ = [
    'ModelWrapper', 'CrossEntropyLoss', 'PolynomialLR'
]