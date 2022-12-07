import os
import sys 

import torch 
import logging 
import argparse

import os.path as osp

from datetime import datetime
from torchsummary import summary
from importlib import import_module
from utils import cvt_cfgPathToDict, build_logger

from builders.builders import build_loaders, build_loss, build_model
from builders.builders import build_optimizer, build_lr_config, build_runner
from pprint import pprint, pformat


def parse_args():

    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')

    args = parser.parse_args()

    return args


def main():
    
    args = parse_args()

    # build config 
    cfg_path = args.config

    cfg = cvt_cfgPathToDict(cfg_path)

    
    # build logger  
    logger, log_path = build_logger(cfg['WORK_DIR'])
    

    # Print paths
    logger.info('Log file is saved at %s' % log_path)
    logger.info(pformat(cfg, width=100))

    # select device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### build data loaders 
    data_loaders = build_loaders(cfg['DATA_LOADERS']) 

    # build model     
    model = build_model(cfg['MODEL'])    
    logger.info(summary(model))
    model.to(device)

    # build optimizer 
    optimizer = build_optimizer(cfg['OPTIMIZER'], model)
    if  cfg['LR_CONFIG']:
        scheduler = build_lr_config(cfg['LR_CONFIG'], optimizer)

    runner = build_runner(cfg['RUNNER'])
    runner.train(
        cfg,
        model, 
        device, 
        logger, 
        optimizer, 
        data_loaders,
        scheduler
    )

    
        
if __name__ == '__main__':
    main()