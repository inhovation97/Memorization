import logging
import wandb
import time
import os
import json
import torch
from collections import OrderedDict
import torch.nn.functional as F


_logger = logging.getLogger('train')

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(model, dataloader, criterion, optimizer, log_interval: int, device: str) -> dict:   
# def train(model, dataloader, criterion, optimizer, log_interval, device):   
    
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    acc_m = AverageMeter()
    losses_m = AverageMeter()
    
    end = time.time()
    
    model.train()
    optimizer.zero_grad()
    for idx, data_dict in enumerate(dataloader):
        data_time_m.update(time.time() - end)
        
        batch_image = data_dict['batch_image'].to(device)
        true_label = data_dict['true_label'].to(device)
        noisy_label = data_dict['noisy_label'].to(device)
        image_path = data_dict['image_path']

        # comparison_label = data_dict['true_label'] == data_dict['noisy_label']
        # true_label_idx = torch.where((comparison_label)==True)[0]
        # false_label_idx = torch.where((comparison_label)==False)[0]

        # predict
        outputs = model(batch_image)
        loss = criterion(outputs, noisy_label)    
        loss.backward()

        # loss update
        optimizer.step()
        optimizer.zero_grad()
        losses_m.update(loss.item())

        # accuracy
        preds = outputs.argmax(dim=1) 
        acc_m.update(noisy_label.eq(preds).sum().item()/noisy_label.size(0), n=noisy_label.size(0)) # eq는 element 별로 비교해서, True of Flase로 출력 (sum으로 맞힌 개수)
        
        batch_time_m.update(time.time() - end)
    
        if (idx+1) % log_interval == 0 and idx != 0: 
            _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) ' 
                        'Acc: {acc.avg:.3%} '
                        'LR: {lr:.3e} '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        idx+1, len(dataloader), 
                        loss       = losses_m, 
                        acc        = acc_m,     
                        lr         = optimizer.param_groups[0]['lr'],
                        batch_time = batch_time_m,
                        rate       = batch_image.size(0) / batch_time_m.val,
                        rate_avg   = batch_image.size(0) / batch_time_m.avg,
                        data_time  = data_time_m))
        end = time.time()
    
    return OrderedDict([('acc',acc_m.avg), ('loss',losses_m.avg)])
        
def test(model, dataloader, criterion, log_interval: int, device: str) -> dict:
    correct = 0
    total = 0
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(dataloader):

            batch_image = data_dict['batch_image'].to(device)
            true_label = data_dict['true_label'].to(device)
            noisy_label = data_dict['noisy_label'].to(device)
            image_path = data_dict['image_path']
            
            # predict
            outputs = model(batch_image)
            
            # loss 
            loss = criterion(outputs, true_label)
            
            # total loss and acc
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += true_label.eq(preds).sum().item()
            total += true_label.size(0)

            if idx % log_interval == 0 and idx != 0: 
                _logger.info('TEST [%d/%d]: Loss: %.3f | Acc: %.3f%% [%d/%d]' % 
                            (idx+1, len(dataloader), total_loss/(idx+1), 100.*correct/total, correct, total))

    return OrderedDict([('acc',correct/total), ('loss',total_loss/len(dataloader))])
                
def fit(
    model, trainloader, validloader, criterion, optimizer, scheduler, 
    epochs: int, savedir: str, data_name: str, log_interval: int, device: str) -> None:

    best_acc = 0
    step = 0

    for epoch in range(epochs):
        _logger.info(f'\nEpoch: {epoch+1}/{epochs}')
        train_metrics = train(model, trainloader, criterion, optimizer, log_interval, device)
        eval_metrics = test(model, validloader, criterion, log_interval, device)

        # wandb
        metrics = OrderedDict(lr=optimizer.param_groups[0]['lr'])
        metrics.update([('train_' + k, v) for k, v in train_metrics.items()])
        metrics.update([('eval_' + k, v) for k, v in eval_metrics.items()])
        wandb.log(metrics, step=step)

        step += 1

        # step scheduler
        if scheduler:
            scheduler.step()

        # checkpoint
        if best_acc < eval_metrics['acc']:

            # save results
            result_path = os.path.join(savedir)
            os.makedirs(result_path, exist_ok=True)
            state = {'best_epoch':epoch, 'best_acc':eval_metrics['acc']}
            json.dump(state, open(os.path.join(result_path, f'best_results.json'),'w'), indent=4)

            # save model
            torch.save(model.state_dict(), os.path.join(result_path, f'best_model.pt'))
            
            _logger.info('Best Accuracy {0:.3%} to {1:.3%}'.format(best_acc, eval_metrics['acc']))

            best_acc = eval_metrics['acc']

        # if int((epochs+1) % 25) == 0 or (epochs+1) == 1:
             # save results
        result_path = os.path.join(savedir)
        os.makedirs(result_path, exist_ok=True)
            # state = {'train_acc':train_metrics['acc'], 'train_loss':train_metrics['loss'],
            #          'eval_acc':eval_metrics['acc'], 'eval_loss':eval_metrics['loss'], }
            # json.dump(state, open(os.path.join(result_path, f'{epoch}_results.json'),'w'), indent=4)

            # save model
        torch.save(model.state_dict(), os.path.join(result_path, f'{epoch}_model.pt'))
            # _logger.info(f'{epoch} Accuracy {0:.3%} to {1:.3%}'.format(best_acc, eval_metrics['acc']))

    _logger.info('Best Metric: {0:.3%} (epoch {1:})'.format(state['best_acc'], state['best_epoch']))