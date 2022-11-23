

from .utils import build_from_cfg

from torch.optim import *
from torch.optim.lr_scheduler import *
from models.lr_schedulers import *

def build_optimizer(cfg, model):

    cfg['params'] = model.parameters()

    return build_from_cfg(cfg, globals())

def build_lr_config(cfg, optimizer):

    cfg['optimizer'] = optimizer

    return build_from_cfg(cfg, globals())