import os
import datetime
import numpy as np
import sys
import re
import argparse
import timm
import random
from collections import defaultdict

import torch
import torchvision
import torchvision.transforms as transforms

            

def transformed(args, mean, std, train=True):

    if train:
      if args.DA == "none":
          transform = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize(mean, std)
          ])

      elif args.DA == "flip_crop":
          transform = transforms.Compose([
              transforms.RandomCrop(32, padding=4),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize(mean, std)
          ])

    else:
      transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
    
    return transform


def make_noisy_label(true_labels, cls_num):

    noisy_label = []
    for t_l in true_labels:
        label_list = np.arange(cls_num)

        # Delete the true label within whole label list
        label_list = np.delete(label_list, int(t_l))
        noisy_label.append(random.choice(label_list))

    noisy_labels = torch.tensor(noisy_label)
    return noisy_labels.cuda()
            
#-------------------------------------------------------------------------------------------------------

# Gradient store
def grad_store(images, targets, model, loss_function):
    model.train()
    model.zero_grad()
    grad_dict = defaultdict(list)

    outputs = model(images)

    loss = loss_function(outputs, targets)
    loss.backward()

    # Extract gradients
    for i, (name, param) in enumerate(model.named_parameters()):
        if ('layer' in name) and ('conv' in name):
            key = name.split('.')[0]
            value = np.array(param.grad.view(-1).clone().detach().cpu())
            grad_dict[key].append(value)
    '''
    grad_dict = 
        {
            'layer1': [grad_vector_list1, grad_veotor_list2, ... ],
            'layer2': [...], 
            ... 
        }
    '''
    return grad_dict


# Calculate mean gradient of all batch
def calc_mean_grad(grad_batch_list):
    
    accumlated_grad_dict = defaultdict(list)
    for i, batch_dict in enumerate(grad_batch_list):
        if i == 0:
            for key, value in batch_dict.items():
                accumlated_grad_dict[key] = value    # value is np.array
        else:
            for (key, v1), (_, v2) in zip(accumlated_grad_dict.items(), batch_dict.items()):
                sum_vector_list = []
                for accumulated_grad_vector, batch_dict_vector in zip(v1, v2):
                    elementwise_sum_vector = accumulated_grad_vector + batch_dict_vector
                    sum_vector_list.append(elementwise_sum_vector)
                    
                accumlated_grad_dict[key] = sum_vector_list

    # Get a mean vector
    mean_grad_dict = defaultdict(list)
    for key, value in accumlated_grad_dict.items():
        mean_grad_dict[key] = [x / len(grad_batch_list) for x in value]

    return mean_grad_dict
#-------------------------------------------------------------------------------------------------------


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.lr * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
