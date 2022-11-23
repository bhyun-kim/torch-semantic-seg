
from .utils import build_from_cfg

from runners.supervised_learning import SupervisedLearner

def build_runner(cfg):
    return build_from_cfg(cfg, globals())
