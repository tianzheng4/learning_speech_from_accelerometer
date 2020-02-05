
import torch
from torch.autograd import Variable
import torch.optim as optim

from torch.nn import functional as F
import numpy as np
import os
import glob
import argparse
from tqdm import tqdm
from feeder_recon import *


from network import *

from metrics import *
from utils import *

import setGPU

def parse_args():
    desc = 'torch implementation of accelerator class'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

    parser.add_argument('--train_audio_path', type=str, default='')
    parser.add_argument('--train_acc_path', type=str, default='')

    parser.add_argument('--test_audio_path', type=str, default='')
    parser.add_argument('--test_acc_path', type=str, default='')

    parser.add_argument('--reconstruct_path', type=str, default='')

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--cuda', type=bool, default=True)

    parser.add_argument('--restart_epoch', type=int, default=500)

    parser.add_argument('--img_col', type=int, default=224)
    parser.add_argument('--img_row', type=int, default=224)

    parser.add_argument('--outfile', type=str, default='log.txt')

    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='')


    return parser.parse_args()


def train(model, optimizer, train_dl, args):
    ## mode
    model.train()

    with tqdm(total=len(train_dl)) as qbar:
        for i, (dataA, dataB) in enumerate(train_dl):
            if args.cuda:
                dataA, dataB = dataA.cuda(), dataB.cuda()

            dataA, dataB = Variable(dataA), Variable(dataB)

            y = model(dataA)

            #print(dataB.max(), dataB.min())

            l1 = torch.nn.L1Loss()
            l2 = torch.nn.MSELoss()

            loss = 100.0*l1(y, dataB)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            qbar.update()

def evaluate(model, test_dl, args):
    summ = []
    model.eval()

    for i, (dataA, dataB) in enumerate(train_dl):
        if args.cuda:
            dataA, dataB = dataA.cuda(), dataB.cuda()

        y = model(dataA)
        ## Calculate some metrics
        summary_batch = {}
        err = (y - dataB).cpu().detach().numpy()
        dataB_numpy = dataB.cpu().detach().numpy()
        y_numpy = y.cpu().detach().numpy()
        dim = np.prod(np.shape(err)[1:])

        l1_absolute_err = np.mean(np.sum(np.abs(err.reshape(-1, dim)), axis=1))
        l1_output = np.mean(np.sum(np.abs(y_numpy.reshape(-1, dim)), axis=1))
        l1_image = np.mean(np.sum(np.abs(dataB_numpy.reshape(-1, dim)), axis=1))

        summary_batch['l1_absolute_err']  = l1_absolute_err
        summary_batch['l1_image']  = l1_image
        summary_batch['l1_output']  = l1_output

        l2_absolute_err = np.mean(np.sqrt(np.sum(np.square(err.reshape(-1, dim)), axis=1)))
        l2_image = np.mean(np.sqrt(np.sum(np.square(dataB_numpy.reshape(-1, dim)), axis=1)))

        summary_batch['l2_absolute_err']  = l2_absolute_err
        summary_batch['l2_image']  = l2_image


        summ.append(summary_batch)

    mean_metrics = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    print(mean_metrics)
    return mean_metrics

def reconstruct(model, args):
    audio_dir = args.test_audio_path + 'subfolder/'
    acc_dir = args.test_acc_path + 'subfolder/'
    reconstruct_dir = args.reconstruct_path
    print(acc_dir)
    for filepath in iter(glob.glob(acc_dir + '*.png')):
        print(filepath)
        filepath = os.path.basename(filepath)

        ## reconstructed img
        img = load_image(acc_dir + filepath)
        recon_img = model(img.cuda().unsqueeze(0))
        recon_img = recon_img.squeeze().cpu().detach().numpy()

        ## original audio img
        img = load_image(audio_dir + filepath)
        img = img[0, :, :]
        img = img.detach().numpy()


        img = np.rint(np.clip((img*255.0), 0, 255)).astype(np.uint8)
        data = img.copy()


        recon_img = np.rint(np.clip((recon_img *255.0), 0, 255)).astype(np.uint8)
        recon_data = recon_img.copy()


        recon_img = np.concatenate((np.expand_dims(recon_img, axis=2),
                                      np.zeros((recon_img.shape[0], recon_img.shape[1], 2), dtype=np.uint8)), axis=2)

        print(recon_img.max())
        recon_img = Image.fromarray(recon_img, 'RGB')
        recon_img.save(reconstruct_dir + filepath)


        ## Test

        recon_img = load_image(reconstruct_dir + filepath).detach().numpy()
        print(recon_img.max())
        err = np.rint(np.clip((recon_img[0, :, :]*255.0), 0, 255)).astype('int') - img.astype('int')
        print(np.sum(np.abs(err)))
        print(np.mean(np.abs(err)))



if __name__ == '__main__':
    args = parse_args()

    model = ImageTransformNet_Spectrogram_Sentence()

    if args.cuda:
        model = model.cuda()
        print('cuda_weights')

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    train_dl = torch.utils.data.DataLoader(dataset=Feeder_Spectrogram(args.train_acc_path,
                                                          args.train_audio_path),
                                            batch_size=4,
                                            shuffle=True,
                                            num_workers=4,
                                            drop_last=True)

    test_dl = torch.utils.data.DataLoader(dataset=Feeder_Spectrogram(args.test_acc_path,
                                                         args.test_audio_path),
                                            batch_size=4,
                                            shuffle=False,
                                            num_workers=4,
                                            drop_last=False)

    outfile = args.save_dir + '/' + args.outfile

    if os.path.isfile(outfile):
        f = open(outfile, 'a')
    else:
        f = open(outfile, 'a')

    if os.path.getsize(outfile) == 0:
        print("epoch\tl1err\tl1image\tl2err\tl2image", file=f, flush=True)

    if args.mode == 'train':

        ## START TRAINING
        best_mse = 1e8
        best_l1 = 1e8


        # optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)

        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)


        for epoch in range(args.num_epochs):

            mean_metrics = evaluate(model, train_dl, args)
            mean_metrics = evaluate(model, test_dl, args)

            print("{}\t{}\t{}\t{}\t{}".format(
                    epoch, mean_metrics['l1_absolute_err'], mean_metrics['l1_image'],
                    mean_metrics['l2_absolute_err'], mean_metrics['l2_image']), file=f, flush=True)


            val_mse = mean_metrics['l2_absolute_err']

            is_best = (val_mse < best_mse)

            if is_best:
                best_mse = val_mse
            print(is_best)
            state = {'epoch': epoch,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict()}
            save_checkpoint(state, epoch, is_best, args.save_dir)

            print('epoch:  ', epoch)
            train(model, optimizer, train_dl, args)

            scheduler.step()

    if args.mode == 'test':

        load_checkpoint(args.load_dir + '/best.pth.tar', model)

        mean_metrics = evaluate(model, train_dl, args)
        mean_metrics = evaluate(model, test_dl, args)

    if args.mode == 'retrain':

        load_checkpoint(args.load_dir + '/last.pth.tar', model)

        mean_metrics = evaluate(model, train_dl, args)
        mean_metrics = evaluate(model, test_dl, args)

        best_l1 = mean_metrics['l1']
        best_mse = mean_metrics['l2']



        optimizer = optim.SGD(model.parameters(), lr=0.1*(0.9**args.restart_epoch), momentum=0.9, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)


        for epoch in range(args.restart_epoch, args.num_epochs):

            mean_metrics = evaluate(model, train_dl, args)
            mean_metrics = evaluate(model, test_dl, args)

            print("{}\t{}\t{}\t{}\t{}".format(
                    epoch, mean_metrics['l1_absolute_err'], mean_metrics['l1_image'],
                    mean_metrics['l2_absolute_err'], mean_metrics['l2_image']), file=f, flush=True)

            val_mse = mean_metrics['l2_absolute_err']

            is_best = (val_mse < best_mse)

            if is_best:
                best_mse = val_mse
            print(is_best)
            state = {'epoch': epoch,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict()}
            save_checkpoint(state, epoch, is_best, args.save_dir)

            print('epoch:  ', epoch)
            train(model, optimizer, train_dl, args)

            scheduler.step()


    if args.mode == 'reconstruct':

        load_checkpoint(args.load_dir + '/best.pth.tar', model)

        if not os.path.isdir(args.reconstruct_path):
            os.mkdir(args.reconstruct_path)

        reconstruct(model, args)
