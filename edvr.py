import os
import argparse
import yaml
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import ShapeNetPart

from models.model import ConsNet
from utils import emd_mixup, emd_mixup_2obj, add_mixup, chamfer_distance, rand_proj, L1_loss, emd_loss, IOStream


seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

def init(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
        os.system('cp ./mixup.yml ./checkpoints/' + args.exp_name)
    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    return io


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args):
    io = init(args)
    train_dataset = ShapeNetPart(partition='trainval', num_points=args.num_points)
    if (len(train_dataset) < 100):
        drop_last = False
    else:
        drop_last = True
    train_loader = DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=drop_last)
    test_loader = DataLoader(ShapeNetPart(partition='test', num_points=args.num_points), 
                            num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    
    seg_num_all = train_loader.dataset.seg_num_all
    seg_start_index = train_loader.dataset.seg_start_index
    
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.model == 'consnet':
        model = ConsNet(args, seg_num_all).to(device)
    else:
        raise Exception("Not implemented")
  
    if args.parallel == True:
        model = nn.DataParallel(model)
        
    print(str(model))

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        
    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
        print('Use COS')
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.7)
        print('Use Step')

    if args.loss == 'l1loss':
        print('Use L1 Loss')
    elif args.loss == 'chamfer':
        print('Use Chamfer Distance')
    else:
        print('Not implemented')
    
    io.cprint('Experiment: %s' % args.exp_name)

    # Train
    min_loss = 100
    io.cprint('Begin to train...')
    for epoch in range(args.epochs):
        io.cprint('=====================================Epoch %d========================================' % epoch)
        io.cprint('*****Train*****')
        # Train
        model.train()
        train_loss = 0
        for i, point in enumerate(train_loader):
            data, label, seg = point
            if epoch < 5:
                lr = 0.00225 * epoch + 0.001
                adjust_learning_rate(opt, lr)

            if args.mixup == 1:
                mixup_data, data1, data2, label1, label2 = emd_mixup(data, label, args.cuda) # (B, N, 3)
            elif args.mixup == 2:
                mixup_data, data1, data2, label1, label2 = emd_mixup_2obj(data, label, args.cuda) # (B, N, 3)
            elif args.mixup == 3:
                if epoch % 2 == 0:
                    mixup_data, data1, data2, label1, label2 = emd_mixup(data, label, args.cuda) # (B, N, 3)
                else:
                    mixup_data, data1, data2, label1, label2 = emd_mixup_2obj(data, label, args.cuda) # (B, N, 3)
            elif args.mixup == 4:
                mixup_data, data1, data2, label1, label2 = add_mixup(data, label, args.cuda)
                
            mixup_data = mixup_data.permute(0, 2, 1) # (B, 3, N)
            batch_size = mixup_data.size()[0]
            
            seg = seg - seg_start_index
            label_one_hot1 = np.zeros((batch_size, 16))
            label_one_hot2 = np.zeros((batch_size, 16))
            for idx in range(batch_size):
                label_one_hot1[idx, label1[idx]] = 1
                label_one_hot2[idx, label2[idx]] = 1
            
            label_one_hot1 = torch.from_numpy(label_one_hot1.astype(np.float32))
            label_one_hot2 = torch.from_numpy(label_one_hot2.astype(np.float32))
            data, label_one_hot1, label_one_hot2, seg = data.to(device), label_one_hot1.to(device), label_one_hot2.to(device), seg.to(device)

            # Project
            proj1 = rand_proj(data1)
            proj2 = rand_proj(data2)
            
            # Train
            opt.zero_grad()
            
            pred1 = model(mixup_data, proj1, label_one_hot1).permute(0, 2, 1) # (B, N, 3)
            pred2 = model(mixup_data, proj2, label_one_hot2).permute(0, 2, 1) # (B, N, 3)

            if args.loss == 'l1loss':
                loss = L1_loss(pred1, data1) + L1_loss(pred2, data2)
            elif args.loss == 'chamfer':
                loss1 = chamfer_distance(pred1, data1) + chamfer_distance(data1, pred1)
                loss2 = chamfer_distance(pred2, data2) + chamfer_distance(data2, pred2)
                loss = loss1 + loss2
            elif args.loss == 'emd':
                loss = emd_loss(pred1, data1) + emd_loss(pred2, data2)

            
            loss.backward()
            
            train_loss = train_loss + loss.item()
            opt.step()

            if (i + 1) % 100 == 0:
                io.cprint('iters %d, tarin loss: %.6f' % (i, train_loss / i))

        io.cprint('Learning rate: %.6f' % (opt.param_groups[0]['lr']))

        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5

        # Test
        if args.valid:
            io.cprint('*****Test*****')
            test_loss = 0
            model.eval()
            for data, label, seg in test_loader:
                with torch.no_grad():
                    if args.mixup == 1:
                        mixup_data, data1, data2, label1, label2 = emd_mixup(data, label) # (B, N, 3)
                    elif args.mixup == 2:
                        mixup_data, data1, data2, label1, label2 = emd_mixup_2obj(data, label) # (B, N, 3)
                    elif args.mixup == 3:
                        if epoch % 2 == 0:
                            mixup_data, data1, data2, label1, label2 = emd_mixup(data, label) # (B, N, 3)
                        else:
                            mixup_data, data1, data2, label1, label2 = emd_mixup_2obj(data, label) # (B, N, 3)
                            
                    mixup_data = mixup_data.permute(0, 2, 1) # (B, 3, N)
                    batch_size = mixup_data.size()[0]

                    seg = seg - seg_start_index
                    label_one_hot1 = np.zeros((batch_size, 16))
                    label_one_hot2 = np.zeros((batch_size, 16))
                    for idx in range(batch_size):
                        label_one_hot1[idx, label1[idx]] = 1
                        label_one_hot2[idx, label2[idx]] = 1

                    label_one_hot1 = torch.from_numpy(label_one_hot1.astype(np.float32))
                    label_one_hot2 = torch.from_numpy(label_one_hot2.astype(np.float32))
                    data, label_one_hot1, label_one_hot2, seg = data.to(device), label_one_hot1.to(device), label_one_hot2.to(device), seg.to(device)

                    proj1 = rand_proj(data1)
                    proj2 = rand_proj(data2)
                    
                    pred1 = model(mixup_data, proj1, label_one_hot1).permute(0, 2, 1) # (B, N, 3)
                    pred2 = model(mixup_data, proj2, label_one_hot2).permute(0, 2, 1) # (B, N, 3)

                    if args.loss == 'l1loss':
                        loss = L1_loss(pred1, data1) + L1_loss(pred2, data2)
                    elif args.loss == 'chamfer':
                        loss1 = chamfer_distance(pred1, data1) + chamfer_distance(data1, pred1)
                        loss2 = chamfer_distance(pred2, data2) + chamfer_distance(data2, pred2)
                        loss = loss1 + loss2
                    elif args.loss == 'emd':
                        loss = emd_loss(pred1, data1) + emd_loss(pred2, data2)

                    test_loss = test_loss + loss.item()
            io.cprint('Train loss: %.6f, Test loss: %.6f' % (train_loss / len(train_loader), test_loss / len(test_loader)))
            cur_loss = test_loss / len(test_loader)
            if cur_loss < min_loss:
                min_loss = cur_loss
                torch.save(model.state_dict(), 'checkpoints/%s/best_%s.pkl' % (args.exp_name, args.exp_name))            
        torch.save(model.state_dict(), 'checkpoints/%s/%s_epoch_%s.pkl' % (args.exp_name, args.exp_name, str(epoch)))    
    torch.save(model.state_dict(), 'checkpoints/%s/%s.pkl' % (args.exp_name, args.exp_name))


def test():
    pass

class Config(dict):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        self.__dict__ = self




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud')
    parser.add_argument('--opt', type=str, default='train.yml', metavar='N',
                        help='config yaml')  
    args = parser.parse_args()
    configyaml = args.opt

    with open(configyaml, 'r', encoding='utf-8') as f:
        config = yaml.safe_load_all(f)
        config = list(config)[0]
    
    config = Config(config)
    
    if config.eval:
        test(config)
    else:
        train(config)
