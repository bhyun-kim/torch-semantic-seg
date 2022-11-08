import os.path as osp

import torch 

from evaluate import evaluate


class SupervisedLearner(object):
    def __init__(self, run_by='epoch'):
        
        self.run_by = run_by 
    
    def train(self, 
             cfg,
             model, 
             device, 
             logger, 
             optimizer, 
             criterion, 
             data_loaders,
             scheduler):

        """
        Args: 
            runner_pack (dict): 
                includes configuration, model, criterion, data_loaders, 
                         device, logger
        """

        logger_interval = cfg['LOGGER']['interval']
        eval_interval = cfg['EVALUATION']['interval']
        checkpoint_interval = cfg['CHECKPOINT']['interval']

        if self.run_by == 'epoch':

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
                    scheduler.step()

                    # print statistics
                    running_loss += loss.item()
                    
                    if i % logger_interval == logger_interval-1:
                        logger.info(
                            f'[Epoch: {epoch + 1:5d}, Iteration: {i + 1:5d}] Loss: {running_loss / logger_interval:.3f}'
                            )
                        running_loss = 0.0

                if i % eval_interval == eval_interval-1:
                    evaluate(model, data_loaders['val'], device, logger)  

                if i % checkpoint_interval == checkpoint_interval-1:
                    save_path = osp.join(cfg['WORK_DIR'], f'checkpoint_epoch_{epoch}.pth')
                    torch.save(model.state_dict(), save_path)

            logger.info('Finished Training')
        
        elif self.run_by == 'iteration': 

            running_loss = 0.0
            generator = iter(data_loaders['train'])

            for i in range(cfg['ITERATION']): # loop over the dataset multiple times

                
                optimizer.zero_grad()
                
                try:
                    data = next(generator)

                except StopIteration: 

                    generator = iter(data_loaders['train'])
                    data = next(generator)
                

                inputs, labels = data['image'], data['segmap']
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                running_loss += loss.item()

                optimizer.step()
                scheduler.step()

                if i % logger_interval == logger_interval-1:
                    logger.info(f'[Iteration: {i + 1:5d}] Loss: {running_loss / logger_interval:.3f}')
                    running_loss = 0.0

                if i % eval_interval == eval_interval-1:
                    evaluate(model, data_loaders['val'], device, logger)  

                if i % checkpoint_interval == checkpoint_interval-1:
                    save_path = osp.join(cfg['WORK_DIR'], f'checkpoint_iter_{i+1}.pth')
                    torch.save(model.state_dict(), save_path)

            logger.info('Finished Training')
        else :
            raise("The variable 'run_by' should be one of ['epoch', iteration']") 



    def eval(self,
             cfg,
             model, 
             device, 
             logger, 
             optimizer, 
             criterion, 
             data_loaders,
             scheduler): 

        evaluate(model, data_loaders['val'], device, logger)  



    