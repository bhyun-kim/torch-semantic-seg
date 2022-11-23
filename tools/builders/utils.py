from copy import deepcopy


def build_from_cfg(cfg, modules):
    """Build from configuration 
    Args: 
        cfg (dict): 
            Configuration dictionary, should have 'type', and other key arguements 

    Returns: 
        module or class 
    """

    name = cfg.pop('type')

    return modules[name](**cfg)
