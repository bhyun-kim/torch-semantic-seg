import torch.nn as nn

from . import OptimRegistry
from torch.optim import Adam as _Adam

@OptimRegistry.register('Adam')
class Adam(_Adam):
    def __init__(self):
        super(Adam, self).__init__()

    
