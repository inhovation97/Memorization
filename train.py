import os
import sys
import argparse
import time
from datetime import datetime
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from data.CIFAR10 import CIFAR10_loader
from data.CIFAR100 import CIFAR100_loader

from model import get_network
from utils import make_noisy_label, calc_mean_grad, grad_store
import random
from tqdm import tqdm

def parse_option():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data', default='CIFAR10', type=str, choices=['CIFAR10', 'CIFAR100'])
  parser.add_argument('--data_path', default=None, type=str, help='Where is your data')
  parser.add_argument('--name', type=str)
  parser.add_argument('--model', type=str, default='resnet18', help='net type')
  parser.add_argument('--pretrain', default=False, help='use pretrained model or not')
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--scheduler", action='store_true', default=False) # default = False
  parser.add_argument("--tracking_grads", action='store_true', default=False)
  
  parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
  parser.add_argument('--val_interval', default=1, type=int, help='validation interval epochs')
  parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 256)')
  parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
  '''
  parser.add_argument('--lr_decay', default=False, type=bool, help='learning rate decay')
  parser.add_argument('--lr_decay_epochs', type=str, default='100,150,180', help='where to decay lr, can be a list')
  parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
  '''
  parser.add_argument('--DA', default='none', type=str, choices=['none', 'flip_crop', 'flip_crop_AA', 'flip_crop_RA'])
  parser.add_argument('--DA_test', default='non', type=str)
  parser.add_argument('--gpu', action='store_true', default=False, help='use gpu or not')
  
  # Important!
  parser.add_argument('--grad_sample_num', default=1024, type=int, help='number of samples to store gradient')
  parser.add_argument('--noisy_comb_len', default=10, type=int, help='number of combinations of noisy labels')
  
  parser.add_argument('--grad_dict_save_path', default="normal_grad_dict.pt", type=str, help='file path for store normal gradient')
  parser.add_argument('--model_save_path', default="noisy_grad_dict.pt", type=str, help='file path for store noisy gradient')
  args = parser.parse_args()

  return args

args = parse_option()

# Save path
normal_grad_epochs_dict = os.path.join(args.grad_dict_save_path, '')

# wandb init
print("wandb init")
def get_timestamp():
    return datetime.now().strftime("%b%d_%H-%M-%S")

wandb.init(
  # Set the project where this run will be logged
  project=f"AAAI 2024 coming soon", 
  name=f"{args.data}_{args.model}_{args.batch_size}_{args.lr}_{get_timestamp()}"
)
wandb.config.update(args)

# Data_loader
if args.data == 'CIFAR10':
  train_loader = CIFAR10_loader(args, is_train=True)
  test_loader = CIFAR10_loader(args, is_train=False)
  cls_num = 10
elif args.data == 'CIFAR100':
  train_loader = CIFAR100_loader(args, is_train=True)
  test_loader = CIFAR100_loader(args, is_train=False)
  cls_num = 100
  
  
# model 
if args.data =='CIFAR10':
  model = get_network(args, class_num=cls_num, pretrain=args.pretrain)
elif args.data =='CIFAR100':
  model = get_network(args, class_num=cls_num, pretrain=args.pretrain)

# continuous training
# if args.continuous_training_epoch !=0:
#   weight_path = os.path.join(args.continuous_training_weight_path, f'{args.continuous_training_epoch}_{args.model}_{args.data}.pt')
#   weights = torch.load()
#   model.load_state_dict(weights)


loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr) # momentum=0.9, weight_decay=5e-4

# scheduler
if args.scheduler:
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
else:
  scheduler = None
    
# Set a seed
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

# Training   
def train(model, epoch):

    epoch_start_time = time.time()
    print('epoch: %d' % epoch)
    model.train()
    
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_index, (images, labels) in enumerate(train_loader):

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum().float().item()
        
        b_idx = batch_index

    print('Train \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('train_Loss: %.3f | train_Acc: %.3f%% (%d/%d)' % (train_loss / (b_idx + 1), 100. * correct / total, correct, total))

    wandb.log({"epoch/train_acc": correct / total * 100, "epoch/trn_loss": train_loss / (b_idx + 1), "epoch": epoch})

    return train_loss / (b_idx + 1), correct / total * 100


# Validation
def validation(model):
    epoch_start_time = time.time()
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, targets) in enumerate(test_loader):
        
        images, targets = images.cuda(), targets.cuda()
            
        outputs = model(images)
        loss = loss_function(outputs, targets)

        val_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        
        b_idx = batch_idx

    print('Validation \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('validation_Loss: %.3f | validation_Acc: %.3f%% (%d/%d)' % (val_loss / (b_idx + 1), 100. * correct / total, correct, total))
    
    wandb.log({"epoch/val_acc": correct / total * 100, "epoch/val_loss": val_loss / (b_idx + 1), "epoch": epoch})

    return val_loss / (b_idx + 1), correct / total * 100


# Start Running

torch_seed(args.seed)

normal_grad_epochs_dict = {}
noisy_grad_epochs_dict = {}

best_epoch_acc = (0, 0)
for epoch in range(args.epochs):

    train_loss, train_accuracy = train(model, epoch)
    if epoch % args.val_interval == 0:
      test_loss, test_accuracy = validation(model)
      
    if test_accuracy >= best_epoch_acc[1]:
      best_epoch_acc = (epoch+1, test_accuracy)
      
    if args.scheduler:
      scheduler.step()

    if args.tracking_grads:
      # Store grad_dic
      normal_grad_list = []
      noisy_grad_batch_list = []
      
      for batch_idx, (images, targets) in tqdm(enumerate(train_loader)):
        if batch_idx == args.grad_sample_num / args.batch_size: # gradient check only : args.grad_sample_num
            break

        images, targets = images.cuda(), targets.cuda()

        normal_grad_batch_dict = grad_store(images, targets, model, loss_function)
        normal_grad_list.append(normal_grad_batch_dict)
        torch.cuda.empty_cache()

        # Get noisy data grads
        noisy_grad_c_list = []
        for _ in range(args.noisy_comb_len):            
            noisy_targets = make_noisy_label(targets, cls_num)
            noisy_grad_c_dict = grad_store(images, noisy_targets, model, loss_function)
            noisy_grad_c_list.append(noisy_grad_c_dict)

        noisy_grad_batch_list.append(calc_mean_grad(noisy_grad_c_list))
        torch.cuda.empty_cache()

      normal_grad_epochs_dict[f'epoch_{epoch}'] = calc_mean_grad(normal_grad_list)
      noisy_grad_epochs_dict[f'epoch_{epoch}'] = calc_mean_grad(noisy_grad_batch_list)
      
    print("-------------------------------------------------------------------------")
print(f"[{best_epoch_acc[0]}/{args.epochs}] Best accuracy: {best_epoch_acc[1]}")
if args.tracking_grads:
  # dict path
  normal_grad_save_path = os.path.join(args.grad_dict_save_path, f'{epoch + 1}_{args.noisy_comb_len}_normal_grad_dict.pt')
  noisy_grad_save_path = os.path.join(args.grad_dict_save_path, f'{epoch + 1}_{args.noisy_comb_len}_noisy_grad_dict.pt')
  model_save_path = os.path.join(args.model_save_path, f'{epoch + 1}_{args.model}_{args.data}.pt')

  # save dict
  torch.save(normal_grad_epochs_dict, normal_grad_save_path)
  torch.save(noisy_grad_epochs_dict, noisy_grad_save_path)
  torch.save(model.state_dict(), model_save_path) 
else:
  model_save_path = os.path.join(args.model_save_path, f'{epoch + 1}_{args.model}_{args.data}.pt')
  torch.save(model.state_dict(), model_save_path) 
    
wandb.finish()
    
