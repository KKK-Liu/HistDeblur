import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, args) -> None:
        super().__init__()
        
        self.args = args
        self.isTrain = args.isTrain
        if args.isTrain:
            self.save_dir = args.ckpt_save_path  # save all the checkpoints to save_dir
        # self.train_loss_names = ['all']
        # self.valid_loss_names = ['all']
        self.train_loss_names = ['all']
        self.valid_loss_names = ['all']
        
        self.model_names = []
        self.nets = []
        self.visual_names = []
        self.optimizers = []
        self.schedulers = []
        
        self.meters = {}
        self.metric = 0  # used for learning rate policy 'plateau'
        self.best_valid_loss = 0.0
        
    @abstractmethod
    def set_input(self):
        pass
    
    @abstractmethod
    def save_network(self):
        pass
    
    @abstractmethod
    def load_network(self):
        pass
    
    
    @abstractmethod
    def update_learning_rate(self):
        pass
    
    @abstractmethod
    def to_cuda(self):
        pass
    
    def meter_init(self):
        for loss_name in self.train_loss_names:
            self.meters['train_loss_{}'.format(loss_name)] = AverageMeter()
        for loss_name in self.valid_loss_names:
            self.meters['valid_loss_{}'.format(loss_name)] = AverageMeter()
            
    def mode(self, type):
        if type == 'train':
            for net in self.nets:
                net.train()
        elif type == 'valid':
            for net in self.nets:
                net.eval()
            
    def get_log_message(self):
        self.log_msg = 'Train loss:{:.8f}, Valid loss:{:.8f}, Best valid loss:{:.8f}'.format(
            self.meters['train_loss_all'].avg,
            self.meters['valid_loss_all'].avg,
            self.best_valid_loss
            )
        return self.log_msg
    
    def epoch_start(self):
        self.reset_meters()
        
    def save_network(self, epoch):
        state = {
            'epoch': epoch,
            'state_dict': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_loss': self.best_valid_loss,
        }
        save_file_name = 'val_best.pth.tar'
        torch.save(state,
            os.path.join(self.save_dir, save_file_name))
        
    def reset_meters(self):
        for _, meter in self.meters.items():
            meter.reset()
            
            
    def update_meters(self, isTrain, n):
        if isTrain:
            for loss_name in self.train_loss_names:
                loss_name = 'train_loss_{}'.format(loss_name)
                self.meters[loss_name].update(getattr(self,loss_name).item(), n=n)
        else:
            for loss_name in self.valid_loss_names:
                loss_name = 'valid_loss_{}'.format(loss_name)
                self.meters[loss_name].update(getattr(self,loss_name).item(), n=n)
       
    
    def get_scalar_dict(self):
        scalar_dict = {}
        
        for loss_name in self.train_loss_names:
            scalar_dict['train_loss_{}'.format(loss_name)] = self.meters['train_loss_{}'.format(loss_name)].avg
        for loss_name in self.valid_loss_names:
            scalar_dict['valid_loss_{}'.format(loss_name)] = self.meters['valid_loss_{}'.format(loss_name)].avg
        
        scalar_dict['lr'] = self.schedulers[0].get_last_lr()[0]
        return scalar_dict
    
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
            
    def epoch_finish(self, epoch):
        save_flag = False
        if epoch == self.args.start_epoch or\
            self.best_valid_loss > self.meters['valid_loss_all'].avg:
            # print("before: bcl:{}, meter:{}".format(self.best_valid_loss , self.meters['valid_loss_all'].avg))
            self.best_valid_loss = self.meters['valid_loss_all'].avg
            save_flag = True
            # print("after: bcl:{}, meter:{}".format(self.best_valid_loss , self.meters['valid_loss_all'].avg))
            
            
        if save_flag:
            self.save_network(epoch)
            
        self.update_learning_rate()
        
        
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0.0
        self.avg = 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count