import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import os

def accuracytop1(output, target, k=1):
    batch_size = target.size(0)

    _, pred = output.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))


    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    return (correct_k.mul_(1.0 / batch_size)).data.item()


def accuracytop2(output, target, k=2):
    batch_size = target.size(0)

    _, pred = output.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    return (correct_k.mul_(1.0 / batch_size)).data.item()

def accuracytop3(output, target, k=3):
    batch_size = target.size(0)

    _, pred = output.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    return (correct_k.mul_(1.0 / batch_size)).data.item()

def accuracytop5(output, target, k=5):
    batch_size = target.size(0)

    _, pred = output.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
    return (correct_k.mul_(1.0 / batch_size)).data.item()


metrics = {
    'accuracytop1': accuracytop1,
    'accuracytop3': accuracytop3,
    'accuracytop5': accuracytop5,
    # could add more metrics such as accuracy for each token type
}

one_metrics = {
    'accuracytop1': accuracytop1,
}
