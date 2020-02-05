   
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
    parser.add_argument('--audio_folder_path', type=str, default='reconstruct/MagAudio/')
    parser.add_argument('--acc_folder_path', type=str, default='reconstruct/MagAcc/')
    
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--cuda', type=bool, default=True)
    
    parser.add_argument('--restart_epoch', type=int, default=500)
    
    parser.add_argument('--img_col', type=int, default=224)
    parser.add_argument('--img_row', type=int, default=224)
    
    parser.add_argument('--outfile', type=str, default='log.txt')
    
    parser.add_argument('--load_dir', type=str, default='models/stylenet')
    parser.add_argument('--save_dir', type=str, default='models/stylenet')
    
    
    return parser.parse_args()
    
    
def train(model, optimizer, train_dl, args):
    ## mode
    model.train()
    
    with tqdm(total=len(train_dl)) as qbar:
        for i, (dataA, dataB, labelA, labelB) in enumerate(train_dl):
            if args.cuda:
                dataA, dataB, labelA, labelB = dataA.cuda(), dataB.cuda(), labelA.cuda(), labelB.cuda()
                
            dataA, dataB, labelA, labelB = Variable(dataA), Variable(dataB), Variable(labelA), Variable(labelB)
            
            
            
            y = model(dataA)
            
            l1 = torch.nn.L1Loss()
            l2 = torch.nn.MSELoss()

            # loss = l2(y, dataB)
            # loss = torch.log((y - dataB)**2).mean()
            loss = l1(y, dataB)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            qbar.update()
            
def evaluate(model, test_dl, args):
    summ = []
    model.eval()
    
    for i, (dataA, dataB, labelA, labelB) in enumerate(test_dl):
        if args.cuda:
            dataA, dataB, labelA, labelB = dataA.cuda(), dataB.cuda(), labelA.cuda(), labelB.cuda()
            
        # dataA, dataB, labelA, labelB = Variable(dataA), Variable(dataB), Variable(labelA), Variable(labelB)
        
       
        y = model(dataA)
        ## Calculate some metrics
        summary_batch = {}
        err = (y - dataB).cpu().detach().numpy()
        dataB_numpy = dataB.cpu().detach().numpy()
        dim = np.prod(np.shape(err)[1:])
        
        l1_absolute_err = np.mean(np.sum(np.abs(err.reshape(-1, dim)), axis=1))
        l1_image = np.mean(np.sum(np.abs(dataB_numpy.reshape(-1, dim)), axis=1))
        
        summary_batch['l1_absolute_err']  = l1_absolute_err
        summary_batch['l1_image']  = l1_image
        
        l2_absolute_err = np.mean(np.sqrt(np.sum(np.square(err.reshape(-1, dim)), axis=1)))
        l2_image = np.mean(np.sqrt(np.sum(np.square(dataB_numpy.reshape(-1, dim)), axis=1)))
        
        summary_batch['l2_absolute_err']  = l2_absolute_err
        summary_batch['l2_image']  = l2_image

        
        summ.append(summary_batch)
    
    mean_metrics = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    print(mean_metrics)
    return mean_metrics
    
def reconstruct(model, args):
    
    for idx in range(10):
        acc_dir = args.acc_folder_path +'/test_dataset/' + str(idx) + '/'
        audio_dir = args.audio_folder_path +'/test_dataset/'+ str(idx) + '/'
        
        for filepath in iter(glob.glob(acc_dir + '*.png')):
            filepath = os.path.basename(filepath)
            img = load_image(acc_dir + filepath)
            print(img.size())
            recon_img = model(img.cuda().unsqueeze(0))
            
            img = load_image(audio_dir + filepath)
            img = img[0, :, :]
            img = img.detach().numpy()
            recon_img = recon_img.squeeze().cpu().detach().numpy()
            
            maximum, minimum = np.maximum(img.max(), recon_img.max()), np.minimum(img.min(), recon_img.min())
            
            img = np.clip(((img - minimum)/(maximum - minimum)*255), 0, 255).astype(np.int8)
            data = img.copy()
            img = Image.fromarray(img)
            img.save('reconstruction_data/' + str(idx) + '/' + filepath)
            
            
            recon_img = np.clip(((recon_img - minimum)/(maximum - minimum)*255), 0, 255).astype(np.int8)
            recon_data = recon_img.copy()
            recon_img = Image.fromarray(recon_img)
            recon_img.save('reconstruction_data/' + str(idx) + '/reconstruct_' + filepath)
            
            print(data - recon_data)
            print(np.abs(data - recon_data).max(), np.abs(data - recon_data).mean())
            
    
if __name__ == '__main__':
    args = parse_args()
    
    model = ImageTransformNet_Spectrogram()
    
    if args.cuda:
        model = model.cuda()
        print('cuda_weights')
        
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    
    train_dl = torch.utils.data.DataLoader(dataset=Feeder_Spectrogram(args.acc_folder_path +'/train_dataset/', 
                                                          args.audio_folder_path +'/train_dataset/'),
                                            batch_size=16,
                                            shuffle=True,
                                            num_workers=4,
                                            drop_last=True)
                                            
    test_dl = torch.utils.data.DataLoader(dataset=Feeder_Spectrogram(args.acc_folder_path +'/test_dataset/', 
                                                         args.audio_folder_path +'/test_dataset/'),
                                            batch_size=16,
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
        
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        
            
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
            ########

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
        
        
        
        optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
          
        # optimizer = optim.SGD(model.parameters(), lr=0.1*(0.9**args.restart_epoch), momentum=0.9, weight_decay=1e-5)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            
        
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
            
            # scheduler.step()
            
            
    if args.mode == 'reconstruct':
    
        load_checkpoint(args.load_dir + '/best.pth.tar', model)
        
        if not os.path.isdir('reconstruction_data'):
            os.mkdir('reconstruction_data')
            
        for idx in range(10):
            if not os.path.isdir('reconstruction_data/'+str(idx)):
                os.mkdir('reconstruction_data/'+str(idx))
                
        reconstruct(model, args)
        

            
        
        
            
        
    
