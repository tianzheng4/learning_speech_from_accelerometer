
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
from feeder_one_class import Feeder


from network import ClassNet
from wide_resnet import Wide_ResNet
from metrics import *
from utils import *
import setGPU

import glob
from feeder_one_class import ImageFeeder


def parse_args():
    desc = 'torch implementation of accelerator class'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--test_folder_path', type=str, default='')

    parser.add_argument('--cuda', type=bool, default=True)

    parser.add_argument('--img_col', type=int, default=224)
    parser.add_argument('--img_row', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=2)

    parser.add_argument('--load_dir', type=str, default='models/densenet_password_hand')
    return parser.parse_args()



def images_evaluate(model, args, outputFile=None):
    summ = []
    model.eval()


    for n in range(args.num_classes):
        test_dir = args.test_folder_path + str(n) + '/'
        output_file = open(test_dir+outputFile, 'w')
        num_imgs = len(glob.glob(test_dir + '*.png'))

        imgfeeder = ImageFeeder(label=n)
        print(num_imgs)
        counter = 0
        for i in range(1, num_imgs+1):
            img_path = test_dir + str(i) + '.png'
            data_batch, label_batch = imgfeeder(img_path)

            if args.cuda:
                data_batch, label_batch = data_batch.cuda(), label_batch.cuda()

            data_batch, label_batch = Variable(data_batch), Variable(label_batch)


            logits = model(data_batch)

            _, pred = logits.data.topk(1, 1, True, True)

            pred = pred.t()
            correct = pred.eq(label_batch.data.view(1, -1).expand_as(pred))

            for j in range(correct.size(0)):
                if correct[j][0].data.item() == 0:
                    print(i)
                    counter += 1
                    print('{}'.format(i), file=output_file, flush=True)

        print(counter)
        output_file.close()


def images_evaluate_logits(model, args, outputFile=None):
    summ = []
    model.eval()
    softmax = torch.nn.Softmax(dim=1)

    for n in range(args.num_classes):
        test_dir = args.test_folder_path + str(n) + '/'
        output_file = open(test_dir+outputFile, 'w')
        num_imgs = len(glob.glob(test_dir + '*.png'))

        imgfeeder = ImageFeeder(label=n)
        print(num_imgs)
        counter = 0
        for i in range(1, num_imgs+1):
            img_path = test_dir + str(i) + '.png'
            data_batch, label_batch = imgfeeder(img_path)

            if args.cuda:
                data_batch, label_batch = data_batch.cuda(), label_batch.cuda()

            data_batch, label_batch = Variable(data_batch), Variable(label_batch)


            logits = model(data_batch)

            prob = softmax(logits).data[0][0].item()
            print(prob)

            print('{}'.format(prob), file=output_file, flush=True)

        output_file.close()

def compare_pwd_others():
    ## password probability

    pwd_file = open(args.test_folder_path+'0/prob.txt', 'r')
    pwd_prob = []
    for p in pwd_file:
        pwd_prob.append(float(p))

    others_file = open(args.test_folder_path+'1/prob.txt', 'r')
    others_idx_file = open(args.test_folder_path+'othersIdx.txt', 'r')

    correct = np.ones((len(pwd_prob), ), dtype=np.float32)
    counter = 0

    for i in others_idx_file:
        sen_idx = int(i) - 1
        others_prob = float(others_file.readline())
        if others_prob >= 0.5:
            counter += 1
        if others_prob >= pwd_prob[sen_idx]:
            correct[sen_idx] = 0.0

    print(np.mean(correct))


    # counter = 0
    # for sen_idx in range(len(pwd_prob)):
    #     if pwd_prob[sen_idx] < 0.5:
    #         counter +=1
    # print(counter)



if __name__ == '__main__':
    args = parse_args()

    model = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16),
                      num_init_features=64, bn_size=4, drop_rate=0.3, num_classes=args.num_classes)

    if args.cuda:
        model = model.cuda()
        print('cuda_weights')


    load_checkpoint(args.load_dir + '/best.pth.tar', model)

    images_evaluate(model, args, outputFile='wrongIndex.txt')
    images_evaluate_logits(model, args, outputFile='prob.txt')
    compare_pwd_others()
