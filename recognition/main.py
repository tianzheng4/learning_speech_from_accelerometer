
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision.models.inception import Inception3
from torchvision.models.densenet import DenseNet
from torch.nn import functional as F
import numpy as np
import os
import argparse
from tqdm import tqdm
from feeder import Feeder, MultiDatasetFeeder


from network import ClassNet

from metrics import *
from utils import *
import setGPU

def parse_args():
    desc = 'torch implementation of accelerator class'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--train_folder_path', type=str, default='')
    parser.add_argument('--test_folder_path', type=str, default='')

    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--save_epoch_interval', type=int, default=20)
    parser.add_argument('--cuda', type=bool, default=True)

    parser.add_argument('--img_col', type=int, default=224)
    parser.add_argument('--img_row', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=10)

    parser.add_argument('--load_dir', type=str, default='model')
    parser.add_argument('--save_dir', type=str, default='model')
    return parser.parse_args()


def train(model, optimizer, train_dl, args):
    ## mode
    model.train()

    with tqdm(total=len(train_dl)) as qbar:
        for i, (data_batch, label_batch) in enumerate(train_dl):
            if args.cuda:
                data_batch, label_batch = data_batch.cuda(), label_batch.cuda()

            data_batch, label_batch = Variable(data_batch), Variable(label_batch)

            logits = model(data_batch)

            y_xent = F.cross_entropy(logits, label_batch)

            loss = y_xent

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            qbar.update()

def evaluate(model, test_dl, args):
    summ = []
    model.eval()

    for i, (data_batch, label_batch) in enumerate(test_dl):
        if args.cuda:
            data_batch, label_batch = data_batch.cuda(), label_batch.cuda()

        data_batch, label_batch = Variable(data_batch), Variable(label_batch)


        logits = model(data_batch)

        y_xent = F.cross_entropy(logits, label_batch)

        summary_batch = {metric:metrics[metric](logits.data, label_batch.data)
                                 for metric in metrics}


        # summary_batch['KLD'] = KLD.sum().data.item()
        summary_batch['entropy']  = y_xent.sum().data.item(),

        summ.append(summary_batch)

    mean_metrics = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    print(mean_metrics)
    return mean_metrics

if __name__ == '__main__':
    args = parse_args()


    model = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16),
                      num_init_features=64, bn_size=4, drop_rate=0.3, num_classes=args.num_classes)


    if args.cuda:
        model = model.cuda()
        print('cuda_weights')

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    train_dl = torch.utils.data.DataLoader(dataset=Feeder(args.train_folder_path),
                                            batch_size=16,
                                            shuffle=True,
                                            num_workers=1,
                                            drop_last=True)

    test_dl = torch.utils.data.DataLoader(dataset=Feeder(args.test_folder_path, is_training=False),
                                            batch_size=16,
                                            shuffle=False,
                                            num_workers=1,
                                            drop_last=False)



    if args.mode == 'train':

        ## START TRAINING
        best_val_acc = 0.0
        best_mse = 1.0
        best_kld = 1e6

        for epoch in range(args.num_epochs):
            if epoch == 0:
                # optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
                optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
            elif epoch == 50:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-2
            elif epoch == 100:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-3

            print('epoch:  ', epoch)
            train(model, optimizer, train_dl, args)

            mean_metrics = evaluate(model, train_dl, args)
            mean_metrics = evaluate(model, test_dl, args)

            val_acc = mean_metrics['accuracytop1']

            is_best = (val_acc > best_val_acc)

            if is_best:
                best_val_acc = val_acc
            print(is_best)
            state = {'epoch': epoch,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'acc': val_acc}
            save_checkpoint(state, epoch, is_best, args.save_dir)


    if args.mode == 'test':


        load_checkpoint(args.load_dir + '/best.pth.tar', model)

        mean_metrics = evaluate(model, train_dl, args)
        mean_metrics = evaluate(model, test_dl, args)
