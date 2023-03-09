import argparse
import time
import os
import torch
import numpy as np
import random

parser = argparse.ArgumentParser()

'''the settings below are used for autodl'''
'''run related'''
parser.add_argument('--name', type=str,default='name', help='name of this run')
parser.add_argument('--ckpt_save_path', type=str,default='./checkpoints/')
parser.add_argument('--dry_run', action='store_true')
parser.add_argument('--isTrain', default=True)
parser.add_argument('--resume', action='store_true')

''' model related '''
parser.add_argument('--model', type=str, default='ATT_Trans_Deblur_Net')
parser.add_argument('--level',type=int, default=3)
parser.add_argument('--style',type=int, default=2)
parser.add_argument('--net',type=str, default='Net_Trans')
parser.add_argument('--mean_shift', type=bool, default=True)
parser.add_argument('--range_of_image', type=float, default=1.0)
parser.add_argument('--amp', type=bool, default=False)
''' dataloader  '''
parser.add_argument('--dataset_name', type=str, default='paired_3')
parser.add_argument('--data_root', type=str, default='')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--pin_memory', type=bool, default=True)


'''log related'''
parser.add_argument('--logfilename', type=str,default='logfile', help='name of the log file')
parser.add_argument('--logfilemode', type=str,default='w', help='mode of the log file')


'''optimizer and scheduler related'''
parser.add_argument('--optimizer', type=str, default='nadam')
parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--end_epoch', type=int, default=300)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-5)

parser.add_argument('--scheduler', type=str, default='multisteplr')
parser.add_argument('--gamma', type=float, default=0.3)
parser.add_argument('--milestones', type=int, nargs='+',default=[100,200,250])
parser.add_argument('--T_max', type=int, default=300)

'''loss function lambda'''
parser.add_argument('--lambda_CM',type=float, default=0.1)
parser.add_argument('--lambda_RR',type=float, default=0.1)
parser.add_argument('--lambda_l1',type=float, default=1.0)
parser.add_argument('--lambda_mse',type=float, default=0.0)
parser.add_argument('--lambda_level1',type=float, default=1.0)
parser.add_argument('--lambda_level2',type=float, default=0.1)
parser.add_argument('--lambda_level3',type=float, default=0.01)
parser.add_argument('--lambda_level4',type=float, default=0.001)

'''other options'''
parser.add_argument('--fixseed', type=bool, default=True)
parser.add_argument('--seed', type=int, default=529)
parser.add_argument('--gpu_id', type=int, default=0)

parser.add_argument('--use_tensorboard', type=bool, default=False)
parser.add_argument('--use_wandb', type=bool, default=True)





def get_arguements():
    args = parser.parse_args()
    # print(args._get_kwargs())
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    assert torch.cuda.is_available(), "how could you do not use cuda???"
    
    args.name = args.name +'-'+ time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    args.ckpt_save_path = os.path.join(args.ckpt_save_path, args.name)
    os.makedirs(args.ckpt_save_path, exist_ok=True)
    
    arg_list = args._get_kwargs()
    with open(os.path.join(args.ckpt_save_path, 'args.txt'), 'w') as f:
        for name, arg in arg_list:
            if isinstance(arg, list):
                arg = ",".join(map(str, arg))
            print("{:>20}:{:<20}".format(name, arg))
            f.write("{:>20}:{:<20}".format(name, arg)+'\n')
            
    if args.fixseed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    
    if args.dry_run:
        args.epoch = 1
        
    return args



if __name__ == '__main__':
    args = get_arguements()
    print(args._get_kwargs())
    arg_list = args._get_kwargs()
    for name, arg in arg_list:
        if isinstance(arg, list):
            arg = ",".join(map(str, arg))
        print("{:>20}:{:<20}".format(name, arg))
