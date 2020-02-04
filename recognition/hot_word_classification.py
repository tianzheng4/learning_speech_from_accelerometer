
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
from wide_resnet import Wide_ResNet
from metrics import *
from utils import *
import setGPU

def parse_args():
    desc = 'torch implementation of accelerator class'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--train_folder_path', type=str, default='/data/zth/accelerate-audio/HotwordSpec/train_dataset/')
    parser.add_argument('--test_folder_path', type=str, default='/data/zth/accelerate-audio/HotwordSpec/test_dataset/')

    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--save_epoch_interval', type=int, default=20)
    parser.add_argument('--cuda', type=bool, default=True)

    parser.add_argument('--img_col', type=int, default=224)
    parser.add_argument('--img_row', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=9)

    parser.add_argument('--load_dir', type=str, default='models/hot_word')
    parser.add_argument('--save_dir', type=str, default='models/hot_word')
    return parser.parse_args()


def train(model, optimizer, train_dl, args):
    ## mode
    model.train()


    with tqdm(total=len(train_dl)) as qbar:
        for i, (data_batch, label_batch) in enumerate(train_dl):
            if args.cuda:
                data_batch, label_batch = data_batch.cuda(), label_batch.cuda()

            data_batch, label_batch = Variable(data_batch), Variable(label_batch)

            # z, mu, sigma, logits = model(data_batch)

            logits = model(data_batch)

            y_xent = F.cross_entropy(logits, label_batch, reduce=False)

            ## reweight the loss
            label_weights = 0.1 + 1.6*(torch.sign(label_batch).data.clone().float())
            # print(label_batch, label_weights)
            weighted_loss = torch.mean(label_weights * y_xent)

            #y_xent = F.cross_entropy(logits, label_batch)

            optimizer.zero_grad()
            weighted_loss.backward()
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

        summary_batch = {metric:one_metrics[metric](logits.data, label_batch.data)
                                 for metric in one_metrics}


        # summary_batch['KLD'] = KLD.sum().data.item()
        summary_batch['entropy']  = y_xent.sum().data.item()

        summ.append(summary_batch)

    mean_metrics = {metric:np.array([x[metric] for x in summ]).mean() for metric in summ[0]}
    print(mean_metrics)
    return mean_metrics

def negative_sample_evaluate(model, test_dl, args):
    summ = []
    model.eval()

    counter = np.zeros((args.num_classes, ), dtype=int)

    for i, (data_batch, label_batch) in enumerate(test_dl):
        if args.cuda:
            data_batch, label_batch = data_batch.cuda(), label_batch.cuda()

        data_batch, label_batch = Variable(data_batch), Variable(label_batch)

        logits = model(data_batch)

        y_xent = F.cross_entropy(logits, label_batch)

        summary_batch = {metric:one_metrics[metric](logits.data, label_batch.data)
                                 for metric in one_metrics}
        summary_batch['entropy']  = y_xent.sum().data.item()
        summ.append(summary_batch)


        predictions = logits.argmax(1).data.cpu().numpy()
        for idx in predictions:
            print(idx)
            counter[idx] += 1.0

    print(counter)
    print(counter/np.sum(counter))

    mean_metrics = {metric:np.array([x[metric] for x in summ]).mean() for metric in summ[0]}
    print(mean_metrics)
    return mean_metrics

def detailed_evaluate(model, test_dl, args):
    summ = []
    model.eval()

    counter = np.zeros((args.num_classes, ), dtype=int)
    tot_count = 0.0

    for i, (data_batch, label_batch) in enumerate(test_dl):
        if args.cuda:
            data_batch, label_batch = data_batch.cuda(), label_batch.cuda()

        data_batch, label_batch = Variable(data_batch), Variable(label_batch)

        logits = model(data_batch)

        y_xent = F.cross_entropy(logits, label_batch)

        summary_batch = {metric:one_metrics[metric](logits.data, label_batch.data)
                                 for metric in one_metrics}
        summary_batch['entropy']  = y_xent.sum().data.item()
        summ.append(summary_batch)


        predictions = logits.argmax(1).data.cpu().numpy()
        label_numpy = label_batch.data.cpu().numpy()
        for idx in range(predictions.shape[0]):
            if predictions[idx] != label_numpy[idx]:
                ## false: not the label, positive: prediction
                counter[predictions[idx]] += 1

            if label_numpy[idx] != 1:
                tot_count += 1

    print(counter, tot_count)
    print(counter/float(tot_count))

    mean_metrics = {metric:np.array([x[metric] for x in summ]).mean() for metric in summ[0]}
    print(mean_metrics)
    return mean_metrics



if __name__ == '__main__':
    args = parse_args()

    # model = ClassNet(img_col=args.img_col, img_row=args.img_row, num_classes=args.num_classes)
    # model = ResNet(Bottleneck, layers=[3, 4, 6, 3], num_classes=args.num_classes)
    # model = Inception3(num_classes=10, aux_logits=False)
    model = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16),
                      num_init_features=64, bn_size=4, drop_rate=0.3, num_classes=args.num_classes)
    # model = Wide_ResNet(16, 8, 0.3, args.num_classes)

    if args.cuda:
        model = model.cuda()
        print('cuda_weights')

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)



    train_dl = torch.utils.data.DataLoader(dataset=Feeder(args.train_folder_path),
                                            batch_size=32,
                                            shuffle=True,
                                            num_workers=4,
                                            drop_last=True)


    test_dl = torch.utils.data.DataLoader(dataset=Feeder(args.test_folder_path, is_training=False),
                                            batch_size=16,
                                            shuffle=False,
                                            num_workers=1,
                                            drop_last=False)

    test_nothot_dl = torch.utils.data.DataLoader(dataset=Feeder(args.test_folder_path, is_training=False, target_index=0),
                                                batch_size=16,
                                                shuffle=False,
                                                num_workers=1,
                                                drop_last=False)

    test_hot_dl = [torch.utils.data.DataLoader(dataset=Feeder(args.test_folder_path, is_training=False, target_index=i),
                                                    batch_size=16,
                                                    shuffle=False,
                                                    num_workers=1,
                                                    drop_last=False) for i in range(1, 8+1)]






    if args.mode == 'train':

        ## START TRAINING
        best_val_acc = 0.0
        best_mse = 1.0
        best_kld = 1e6


        for epoch in range(args.num_epochs):
            if epoch == 0:
                # optimizer = optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-5)
                optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
            elif epoch == 50:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-2
            elif epoch == 100:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-3
                # optimizer = optim.Adam(model.parameters(), lr=2e-4)
            # elif epoch == 100:
                # optimizer = optim.Adam(model.parameters(), lr=2e-4)
            # elif epoch == 200:
                # optimizer = optim.Adam(model.parameters(), lr=1e-4)
            print('epoch:  ', epoch)
            train(model, optimizer, train_dl, args)

            mean_metrics = evaluate(model, train_dl, args)
            mean_metrics = evaluate(model, test_dl, args)

            nothot_acc = evaluate(model, test_nothot_dl, args)['accuracytop1']
            hot_acc = 0.0
            for i in range(len(test_hot_dl)):
                hot_acc += evaluate(model, test_hot_dl[i], args)['accuracytop1']
            hot_acc = hot_acc/float(len(test_hot_dl))
            print(nothot_acc, hot_acc)

            val_acc = 0.5*nothot_acc + 0.5*hot_acc

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

        # mean_metrics = evaluate(model, train_dl, args)
        # mean_metrics = evaluate(model, test_dl, args)
        # mean_metrics = evaluate(model, test_sensitive_dl, args)
        # mean_metrics = evaluate(model, test_insensitive_dl, args)
        detailed_evaluate(model, test_dl, args)
        hot_acc = 0.0
        for i in range(len(test_hot_dl)):
            hot_acc += evaluate(model, test_hot_dl[i], args)['accuracytop1']
