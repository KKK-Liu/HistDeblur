import torch.nn as nn
import torch.utils.data
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ConstantLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
from .base_model_new import BaseModel
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp.grad_scaler import GradScaler

from functools import partial

import torchvision.transforms.functional as f
import os
import numpy as np

from .Network import Net, Net_concat, Net_noLSTM, Net_concat_small, Net_concat_wide

class ATT_Deblur_Net(nn.Module, BaseModel):
    def __init__(self, args) -> None:
        super(ATT_Deblur_Net, self).__init__()
        BaseModel.__init__(self, args)
        self.level = args.level
        if args.net == 'Net':
            self.net = Net(level = args.level, style=args.style, mean_shift = args.mean_shift, range_of_image = args.range_of_image)
        elif args.net == 'Net_concat':
            self.net = Net_concat(level = args.level, style=args.style, mean_shift = args.mean_shift, range_of_image = args.range_of_image)
        elif args.net == 'Net_concat_small':
            self.net = Net_concat_small(level = args.level, style=args.style, mean_shift = args.mean_shift, range_of_image = args.range_of_image)
        elif args.net == 'Net_concat_wide':
            self.net = Net_concat_wide(level = args.level, style=args.style, mean_shift = args.mean_shift, range_of_image = args.range_of_image)
        else:
            raise NotImplementedError('{} is not implemented'.format(args.net))
        self.nets.append(self.net)
        
        self.isTrain = self.args.isTrain
        if self.isTrain:
            if args.optimizer == 'adam':
                self.optimizer = torch.optim.Adam( 
                    self.net.parameters(),
                    lr = args.lr,
                    weight_decay = args.weight_decay,
                    amsgrad=True
                )
            elif args.optimizer == 'sgd':
                self.optimizer =optim.SGD(
                    self.net.parameters(), 
                    lr=args.lr,
                    momentum=args.momentum, 
                    nesterov=True, 
                    weight_decay=args.weight_decay)
            elif args.optimizer == 'nadam':
                self.optimizer =optim.NAdam(
                    self.net.parameters(), 
                    lr=args.lr,
                    weight_decay=args.weight_decay)
                
            # self.scaler = GradScaler()
            # optim.NAdam
            if args.scheduler == 'multisteplr':
                self.scheduler = MultiStepLR(self.optimizer, args.milestones, args.gamma)
            elif args.scheduler == 'constantlr':
                self.scheduler = MultiStepLR(self.optimizer, args.milestones, 1.0)
            elif args.scheduler == 'cosineannealinglr':
                self.scheduler = CosineAnnealingLR(self.optimizer,T_max=args.T_max)
            
            self.schedulers.append(self.scheduler)
            
            self.upsample_fn = partial(torch.nn.functional.interpolate, mode='bilinear')
            self.downsample_fn = partial(torch.nn.functional.interpolate, mode='bilinear')
            
            self.lambda_CM = args.lambda_CM
            self.lambda_RR = args.lambda_RR
            self.lambda_l1 = args.lambda_l1
            self.lambda_mse = args.lambda_mse
            
            self.lambda_level1 = min(args.lambda_level1, max(args.level-0,0)) 
            self.lambda_level2 = min(args.lambda_level2, max(args.level-1,0))
            self.lambda_level3 = min(args.lambda_level3, max(args.level-2,0))
            self.lambda_level4 = min(args.lambda_level4, max(args.level-3,0))
            
            self.train_loss_names += ['l1','l1_1','l1_2','l1_3','l1_4',
                                      'mse','mse_1','mse_2','mse_3','mse_4',
                                      'consistency_confidence','consistency']
            self.valid_loss_names += ['l1','l1_1','l1_2','l1_3','l1_4','consistency_confidence','consistency'
                                      ,'mse','mse_1','mse_2','mse_3','mse_4','psnr']
            self.meter_init()
        else:
            self.result_save_root = args.result_save_root 
            os.makedirs(os.path.join(self.result_save_root, 'image'), exist_ok=True)
            self.visual_names = ['restored_images_1','attention_1']
            self.eval_losses = []
            self.image_names = [] 
        self.loss_function_mse = nn.MSELoss()
        self.loss_function_l1 = nn.L1Loss()

        print('ATT Deblur Model is created')
        
    def get_visuals(self):
        with torch.no_grad():
            with autocast(enabled=self.args.amp):
                self.forward()
                
        self.restored_images_1 = self.restored_images_1.clamp(0,1)
        self.restored_images_2 = self.restored_images_2.clamp(0,1)
        self.restored_images_3 = self.restored_images_3.clamp(0,1)
        self.restored_images_4 = self.restored_images_4.clamp(0,1)
        
        self.attention_1 = torch.abs(self.attention_1)
        self.attention_2 = torch.abs(self.attention_2)
        self.attention_3 = torch.abs(self.attention_3)
        self.attention_4 = torch.abs(self.attention_4)
                
    def save_visuals(self):
        for visual_name in self.visual_names:
            visual_batch = getattr(self,visual_name).detach().cpu()
            for image, name in zip(visual_batch, self.paths):
                image = f.to_pil_image(image)
                save_file_name = os.path.join(
                    self.result_save_root,
                    'image',
                    '{}_{}.png'.format(name.replace('.png',''),visual_name)
                )
                image.save(save_file_name)
        
        
    def eval_visuals(self, metrics:dict):
        self.image_names += self.paths
        
        for image_restored, image_sharp in zip(self.restored_images_1, self.sharp_images_1):
            this_losses = []
            for _, metric_function in metrics.items():
                this_losses.append(metric_function(image_restored.unsqueeze(0), image_sharp.unsqueeze(0)).data)
            self.eval_losses.append(this_losses)

    def eval_result_save(self, metrics:dict):
        with open(os.path.join(self.result_save_root, 'metric values.csv'), 'w') as f:
            line = ','.join(['image name']+list(metrics.keys()))+'\n'
            f.write(line)
            for this_losses, image_name in zip(self.eval_losses, self.image_names):
                line = ','.join([image_name]+list(map(str, this_losses))) + '\n'
                f.write(line)


        self.eval_losses = np.array(self.eval_losses)
        np.save(os.path.join(self.result_save_root,'eval_losses.npy'), self.eval_losses)
        
        losses_item = np.mean(self.eval_losses, axis=0)

        with open(os.path.join(self.result_save_root, 'eval_result.txt'), 'w') as f:
            for metric, value in zip(metrics.keys(), losses_item):
                f.write("{:>20}:{:<20}\n".format(metric, value))
                
        with open(os.path.join(self.result_save_root, 'eval_result.txt'), 'r') as f:
            for line in f.readlines():
                print(line)
            
    def to_cuda(self):
        self.net = self.net.cuda()
        self.loss_function_mse = self.loss_function_mse.cuda()
        self.loss_function_l1 = self.loss_function_l1.cuda()
        return self
        
    def set_input(self, data):
        if self.isTrain:
            self.blur_images_1, self.blur_images_2, self.blur_images_3,self.blur_images_4,\
                self.sharp_images_1, self.sharp_images_2, self.sharp_images_3, self.sharp_images_4 = data
        else:
            self.blur_images_1, self.blur_images_2, self.blur_images_3,self.blur_images_4,\
                self.sharp_images_1, self.sharp_images_2, self.sharp_images_3,self.sharp_images_4, self.paths = data
                
        self.blur_images_4  = self.blur_images_4.cuda(non_blocking = True)
        self.blur_images_3  = self.blur_images_3.cuda(non_blocking = True)
        self.blur_images_2  = self.blur_images_2.cuda(non_blocking = True)
        self.blur_images_1  = self.blur_images_1.cuda(non_blocking = True)
        
        self.sharp_images_1 = self.sharp_images_1.cuda(non_blocking = True)
        self.sharp_images_2 = self.sharp_images_2.cuda(non_blocking = True)
        self.sharp_images_3 = self.sharp_images_3.cuda(non_blocking = True)
        self.sharp_images_4 = self.sharp_images_4.cuda(non_blocking = True)
        
    def forward(self):
        self.restored_images_1, self.restored_images_2, self.restored_images_3,self.restored_images_4,\
        self.attention_1, self.attention_2, self.attention_3, self.attention_4 = self.net(self.blur_images_1, self.blur_images_2, self.blur_images_3, self.blur_images_4)        
        
    def train_step(self):
        self.optimizer.zero_grad()
        with autocast(enabled=self.args.amp):
            self.forward()
            
            self.train_loss_l1_1 = self.loss_function_l1(self.restored_images_1, self.sharp_images_1)
            self.train_loss_l1_2 = self.loss_function_l1(self.restored_images_2, self.sharp_images_2)
            self.train_loss_l1_3 = self.loss_function_l1(self.restored_images_3, self.sharp_images_3)
            self.train_loss_l1_4 = self.loss_function_l1(self.restored_images_4, self.sharp_images_4)
            
            self.train_loss_l1 =    self.lambda_level1 * self.train_loss_l1_1 +\
                                    self.lambda_level2 * self.train_loss_l1_2 +\
                                    self.lambda_level3 * self.train_loss_l1_3 +\
                                    self.lambda_level4 * self.train_loss_l1_4
            
            self.train_loss_mse_1 = self.loss_function_mse(self.restored_images_1, self.sharp_images_1)
            self.train_loss_mse_2 = self.loss_function_mse(self.restored_images_2, self.sharp_images_2)
            self.train_loss_mse_3 = self.loss_function_mse(self.restored_images_3, self.sharp_images_3)
            self.train_loss_mse_4 = self.loss_function_mse(self.restored_images_4, self.sharp_images_4)
            
            self.train_loss_mse =   self.lambda_level1 * self.train_loss_mse_1 +\
                                    self.lambda_level2 * self.train_loss_mse_2 +\
                                    self.lambda_level3 * self.train_loss_mse_3 +\
                                    self.lambda_level4 * self.train_loss_mse_4
            
            self.train_loss_consistency_confidence =\
                self.lambda_level4 * self.loss_function_l1(self.attention_4,self.downsample_fn(self.attention_3, (28 ,28 ))) +\
                self.lambda_level3 * self.loss_function_l1(self.attention_3,self.downsample_fn(self.attention_2, (56 ,56 ))) +\
                self.lambda_level2 * self.loss_function_l1(self.attention_2,self.downsample_fn(self.attention_1, (112,112)))
                
            self.train_loss_consistency = \
                self.lambda_level4 * self.loss_function_l1(self.restored_images_4, self.downsample_fn(self.restored_images_3, (28 ,28 ))) +\
                self.lambda_level3 * self.loss_function_l1(self.restored_images_3, self.downsample_fn(self.restored_images_2, (56 ,56 ))) +\
                self.lambda_level2 * self.loss_function_l1(self.restored_images_2, self.downsample_fn(self.restored_images_1, (112,112)))
            
            
            self.train_loss_all = self.lambda_l1 * self.train_loss_l1
                
            if not (torch.isnan(self.train_loss_mse).any() or torch.isinf(self.train_loss_mse).any()):
                self.train_loss_all += self.lambda_mse * self.train_loss_mse
                
            if not (torch.isnan(self.train_loss_consistency).any() or torch.isinf(self.train_loss_consistency).any()):
                self.train_loss_all += self.lambda_RR * self.train_loss_consistency/max(self.level-1, 1)
                
            if not (torch.isnan(self.train_loss_consistency_confidence).any() or torch.isinf(self.train_loss_consistency_confidence).any()):
                self.train_loss_all += self.lambda_CM * self.train_loss_consistency_confidence/max(self.level-1, 1)
                

        if torch.isnan(self.train_loss_mse).any() or torch.isinf(self.train_loss_mse).any():
            raise RuntimeError('nan!!!')
        
        # self.scaler.scale(self.train_loss_all).backward()
        self.train_loss_all.backward()
        # for group in self.optimizer.param_groups:
        #     for param in group["params"]:
        #         if param.grad is not None:
        #             param.grad.data.clamp_(-0.05, 0.05)
                
        if self.args.net == 'Net':
            torch.nn.utils.clip_grad_norm_(self.net.convlstm.parameters(),3)
        # nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=3, norm_type=2)
        
        # self.scaler.step(self.optimizer)
        # self.scaler.update()
        self.optimizer.step()

        self.update_meters(True, self.blur_images_3.size(0))
        
    def valid_step(self):
        with torch.no_grad():
            with autocast(enabled=self.args.amp):
                self.forward()
                
                self.restored_images_1 = self.restored_images_1.clamp(0,1)
                self.restored_images_2 = self.restored_images_2.clamp(0,1)
                self.restored_images_3 = self.restored_images_3.clamp(0,1)
                self.restored_images_4 = self.restored_images_4.clamp(0,1)
                
                self.valid_loss_l1_1 = self.lambda_level1 * self.loss_function_l1(self.restored_images_1, self.sharp_images_1)
                self.valid_loss_l1_2 = self.lambda_level2 * self.loss_function_l1(self.restored_images_2, self.sharp_images_2)
                self.valid_loss_l1_3 = self.lambda_level3 * self.loss_function_l1(self.restored_images_3, self.sharp_images_3)
                self.valid_loss_l1_4 = self.lambda_level4 * self.loss_function_l1(self.restored_images_4, self.sharp_images_4)
                self.valid_loss_l1 = self.valid_loss_l1_1 + self.valid_loss_l1_2 + self.valid_loss_l1_3 + self.valid_loss_l1_4

                
                self.valid_loss_mse_1 = self.lambda_level1 * self.loss_function_mse(self.restored_images_1, self.sharp_images_1)
                self.valid_loss_mse_2 = self.lambda_level2 * self.loss_function_mse(self.restored_images_2, self.sharp_images_2)
                self.valid_loss_mse_3 = self.lambda_level3 * self.loss_function_mse(self.restored_images_3, self.sharp_images_3)
                self.valid_loss_mse_4 = self.lambda_level4 * self.loss_function_mse(self.restored_images_4, self.sharp_images_4)
                self.valid_loss_mse = self.valid_loss_mse_1 + self.valid_loss_mse_2 + self.valid_loss_mse_3 + self.valid_loss_mse_4
                
                self.valid_loss_psnr = -10 * torch.log10(self.valid_loss_mse_1 + 1e-8)

                self.valid_loss_consistency_confidence =\
                    self.lambda_level4 * self.loss_function_l1(self.attention_4,self.downsample_fn(self.attention_3, (28 ,28 ))) +\
                    self.lambda_level3 * self.loss_function_l1(self.attention_3,self.downsample_fn(self.attention_2, (56 ,56 ))) +\
                    self.lambda_level2 * self.loss_function_l1(self.attention_2,self.downsample_fn(self.attention_1, (112,112)))
                    
                self.valid_loss_consistency = \
                    self.lambda_level4 * self.loss_function_l1(self.restored_images_4, self.downsample_fn(self.restored_images_3, (28 ,28 ))) +\
                    self.lambda_level3 * self.loss_function_l1(self.restored_images_3, self.downsample_fn(self.restored_images_2, (56 ,56 ))) +\
                    self.lambda_level2 * self.loss_function_l1(self.restored_images_2, self.downsample_fn(self.restored_images_1, (112,112)))
                
                self.valid_loss_all = self.valid_loss_mse_1
            
            self.update_meters(False, self.blur_images_3.size(0))
        
        
    def load_network(self):
        ckpt = torch.load(self.args.load_ckpt_path)
        if self.isTrain:
            self.net.load_state_dict(ckpt['state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
        else:
            self.net.load_state_dict(ckpt['state_dict'])
            
    def get_and_save_visual_results(self):
        with torch.no_grad():
            with autocast():
                self.forward()
                
                for img_tensor, name in zip(self.restored_images_1, self.paths):
                    img_rgb = f.to_pil_image(img_tensor)
                    img_rgb.save(os.path.join(self.result_save_root, name))
        
    def resume(self):
        self.load_network()
        
        
if __name__ == '__main__':
    ...
