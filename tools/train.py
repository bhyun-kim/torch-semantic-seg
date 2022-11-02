import sys 
import os
import os.path as osp

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import argparse
import logging 
import json

import torch 
import torchvision as tv 

from importlib import import_module
from tqdm import tqdm

from utils import * 
from evaluate import * 
from builders.builders import *  

from datetime import datetime

def parse_args():

    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')

    args = parser.parse_args()

    return args


def main():
    from pprint import pprint
    args = parse_args()

    # build config 
    _cfg = args.config

    abs_path = osp.abspath(_cfg)

    sys.path.append(osp.split(abs_path)[0])
    _mod = import_module(osp.split(abs_path)[1].replace('.py', ''))

    cfg = cvt_moduleToDict(_mod)

    # build logger  
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    os.makedirs(cfg['WORK_DIR'], exist_ok=True)
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_file = cfg['WORK_DIR'] + f'/{current_time}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print paths
    logger.info('Log file is %s' % log_file)
    
    logger.info(pprint(cfg, width=100))

    # select device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### build data loaders 
    data_loaders = build_loaders(cfg['DATA_LOADERS']) 

    # build loss 
    criterion = build_loss(cfg['LOSS'])
    criterion.weight_to(device)

    # build model     
    model = build_model(cfg['MODEL'])    
    model.to(device)

    # build optimizer 
    optimizer = build_optimizer(cfg['OPTIMIZER'], model)

    # build runner -----------------------------------------------------
    for epoch in range(cfg['EPOCHS']): # loop over the dataset multiple times

        running_loss = 0.0
        
        for i, data in enumerate(data_loaders['train'], 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data['image'], data['segmap']

            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:    # print every 50 mini-batches
                logger.info(f'[Epoch: {epoch + 1}, Iteration: {i + 1:5d}] Loss: {running_loss / 50:.3f}')
                running_loss = 0.0

        # evaluate(model, data_loaders['val'], device, 19) 
        if epoch % 5 == 0:
            evaluate(model, data_loaders['val'], device, logger)  
            save_path = osp.join(cfg['WORK_DIR'], f'checkpoint_{epoch}.pth')
            torch.save(model.state_dict(), save_path)

        

    logger.info('Finished Training')
        
if __name__ == '__main__':
    main()