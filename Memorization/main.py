import numpy as np
import os
import random
import wandb
import glob

import torch
import argparse
import logging
import yaml
import timm
from timm import create_model
from data.noisy_dataset import Noisy_dataset

from train import fit
from torch.utils.data import DataLoader
from log import setup_default_logging
import albumentations
from albumentations.pytorch import ToTensorV2 

_logger = logging.getLogger('train')

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

def run(cfg):
    # make save directory
    savedir = os.path.join(cfg['RESULT']['savedir'], cfg['EXP_NAME'])
    os.makedirs(savedir, exist_ok=True)

    setup_default_logging(log_path=os.path.join(savedir,'log.txt'))
    torch_seed(cfg['SEED'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

        # Prearing Dataset
    if cfg['DATASET']['dataname']  == 'cifar100':
        mean = [0.507, 0.487, 0.441]
        stdv = [0.267, 0.256, 0.276]
        num_classes=100
    elif cfg['DATASET']['dataname']  == 'cifar10':
        mean = [0.491, 0.482, 0.447]
        stdv = [0.247, 0.243, 0.262]
        num_classes=10


    if cfg['MODEL']=='vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        model.to(device)

    else:
        model = create_model(cfg['MODEL'], num_classes=num_classes, pretrained=False)
        model.to(device)
        
    _logger.info('# of params: {}'.format(np.sum([p.numel() for p in model.parameters()])))

    # augmentations
    # train_augs = albumentations.Compose([       albumentations.Resize(224, 224),   
    #                                         albumentations.OneOf([
    #                                                             albumentations.HorizontalFlip(p=0.5), 
    #                                                             albumentations.RandomRotate90(p=0.5), 
    #                                                             albumentations.VerticalFlip(p=0.5)], 
    #                                                         p=1),

    #                                         albumentations.OneOf([
    #                                                             albumentations.MotionBlur(p=0.5), 
    #                                                             albumentations.RandomContrast(p=0.5),
    #                                                             albumentations.RandomBrightnessContrast(p=0.5), 
    #                                                             albumentations.GaussNoise(p=0.5)], 
    #                                                         p=1),

    #                                         albumentations.Normalize(mean = mean, std = stdv),
    #                                         ToTensorV2() ])


    test_augs = albumentations.Compose([        
                                                albumentations.Resize(224, 224),  
                                                albumentations.Normalize(mean = mean, std = stdv),
                                                ToTensorV2()])


    train_json_path = cfg['DATASET']['train_json_path']
    test_json_path = cfg['DATASET']['test_json_path']

    ## dataset
    trainset = Noisy_dataset(
        json_path=train_json_path,
        transform=test_augs,
        )

    testset = Noisy_dataset(
        json_path=test_json_path,
        transform=test_augs,
        )
    # load dataloader
    trainloader = DataLoader(dataset=trainset, batch_size=cfg['TRAINING']['batch_size'], shuffle=True)
    validloader = DataLoader(dataset=testset, batch_size=cfg['TRAINING']['test_batch_size'], shuffle=False)
    # testloader = DataLoader(dataset=testset, batch_size=cfg['TRAINING']['test_batch_size'], shuffle=False)

    # set training
    criterion = torch.nn.CrossEntropyLoss()

    if cfg['OPTIMIZER']['opt_name'].lower() == 'sgd':
            optimizer = torch.optim.SGD(
            params       = model.parameters(), 
            lr           = cfg['OPTIMIZER']['lr'], 
        )
    elif cfg['OPTIMIZER']['opt_name'].lower() == 'adam':
            optimizer = torch.optim.Adam(
            params       = model.parameters(), 
            lr           = cfg['OPTIMIZER']['lr'], 
        )

    # scheduler
    if cfg['TRAINING']['use_scheduler']:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['TRAINING']['epochs'])
    else:
        scheduler = None

    # initialize wandb
    if cfg['WANDB']:
        wandb.init(name=cfg['EXP_NAME'], 
                project='Memorization', 
                config=cfg)

    # fitting model
    print('Start the training')
    fit(model        = model, 
        trainloader  = trainloader, 
        validloader  = validloader, 
        criterion    = criterion, 
        optimizer    = optimizer, 
        scheduler    = scheduler,
        epochs       = cfg['TRAINING']['epochs'], 
        savedir      = savedir,
        data_name    = cfg['DATASET']['dataname'],
        log_interval = cfg['TRAINING']['log_interval'],
        device       = device)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='DF_detection')
    parser.add_argument('--default_setting', type=str, default=None, help='exp config file')    
    parser.add_argument('--modelname', type=str, default='resnet18', help='model name')
    parser.add_argument('--dataname', type=str, default=None, help='data name')
    parser.add_argument('--wandb', action='store_false', help='Trun off wandb') # default True
    # parser.add_argument('--exp_title', type=str, default=None, help='data name')

    # Data arguments
    parser.add_argument('--noise_ratio', type=float, help='mode')
    parser.add_argument('--strict_noise', action='store_false', help='strict means never allowcating same label') # default True

    args = parser.parse_args()

    # config -> default setting (batch size, lr, optimizer, etc..)
    cfg = yaml.load(open(args.default_setting,'r'), Loader=yaml.FullLoader)
    
    # cfg is a dict in a dict
    cfg['MODEL'] = args.modelname
    cfg['DATASET']['dataname'] = args.dataname  
    cfg['WANDB'] = args.wandb
    
    cfg['DATASET']['noise_ratio'] = args.noise_ratio
    cfg['DATASET']['strict_noise'] = args.strict_noise
    cfg['EXP_NAME'] = f"{args.modelname}-{args.dataname}_noise_ratio_{args.noise_ratio}_img224"
    run(cfg)