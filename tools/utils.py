from importlib import import_module
from datetime import datetime

import os 
import sys 
import logging


import os.path as osp


def cvt_pathToModule(file_path):
    """Convert path (string) to module form.

    Args :
        file_path (str) : file path written in nomal path form
    
    Returns :
        module_form (str) : file path in module form (i.e. matplotlib.pyplot)
    
    """
    
    file_path = file_path.replace('/', '.')
    module_form  = file_path.replace('.py', '')
    
    return module_form

def cvt_moduleToDict(mod) :
    """
    Args : 
        mod (module)  
    
    Returns :
        cfg (dict)
    
    """
    cfg = {
        name: value
        for name, value in mod.__dict__.items()
        if not name.startswith('__')
        }
    
    return cfg


def cvt_cfgPathToDict(path):
    """Convert configuration path to dictionary to
    Args: 
        path (str)

    Returns: 
        cfg (dict)
    """

    abs_path = osp.abspath(path)

    sys.path.append(osp.split(abs_path)[0])
    _mod = import_module(osp.split(abs_path)[1].replace('.py', ''))

    return cvt_moduleToDict(_mod)



def build_logger(work_dir):
    """Build Logger 

    Args: 
        work_dir (str)

    Returns: 
        logger (logging.logger)
        log_path (str)

    """
    # build logger  

    os.makedirs(work_dir, exist_ok=True)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    
    log_path = work_dir + f'/{current_time}.log'
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_path
