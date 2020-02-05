# sys
import pickle

# torch
import torch
# from torch.autograd import Variable
import torchvision
from torchvision import transforms
import numpy as np


class Feeder(torch.utils.data.Dataset):

    def __init__(self,
                 input_folder_path,
                 output_folder_path,
                 is_training = True,
                 normalization=True,
                 mmap=False,
                 lw_factor = 3,
                 ):

        self.input_dataset = torchvision.datasets.ImageFolder(input_folder_path)
        self.output_dataset = torchvision.datasets.ImageFolder(output_folder_path)

        self.is_training = is_training



        self.input_transform = transforms.Compose([
                                             #transforms.Resize(size=(lw_factor*128, 128), interpolation=3),
                                             transforms.ToTensor(),
                                             ])

        self.output_transform = transforms.Compose([
                                             #transforms.Resize(size=(w_factor*128, 128), interpolation=3),
                                             transforms.ToTensor(),
                                             ])



    def __len__(self):
        return len(self.input_dataset)

    def __iter__(self):
        return self

    def __getitem__(self, index):

        dataA, labelA = self.input_dataset[index]
        dataB, labelB = self.output_dataset[index]

        if self.is_training:
            dataA = self.input_transform(dataA)
            dataA = dataA[:,0:lw_factor*128,:]
            dataA = dataA.contiguous()
            dataA = dataA.view(lw_factor*3, 128, 128)

            dataB = self.output_transform(dataB)
            dataB = dataB[:,0:lw_factor*128,:]
            return dataA-dataA[0, 0, 0], dataB-dataB[0, 0, 0], labelA, labelB
        else:
            dataA = self.input_transform(dataA)
            dataA = dataA[:,0:lw_factor*128,:]
            dataA = dataA.contiguous()
            dataA = dataA.view(lw_factor*3, 128, 128)

            dataB = self.output_transform(dataB)
            dataB = dataB[:,0:lw_factor*128,:]
            return dataA-dataA[0, 0, 0], dataB-dataB[0, 0, 0], labelA, labelB



class Feeder_Spectrogram(torch.utils.data.Dataset):

    def __init__(self,
                 input_folder_path,
                 output_folder_path,
                 is_training = True,
                 normalization=True,
                 mmap=False,
                 lw_factor = 3,
                 ):

        self.input_dataset = torchvision.datasets.ImageFolder(input_folder_path)
        self.output_dataset = torchvision.datasets.ImageFolder(output_folder_path)

        self.is_training = is_training



        self.input_transform = transforms.Compose([
                                             transforms.ToTensor(),
                                             ])

        self.output_transform = transforms.Compose([
                                             transforms.ToTensor(),
                                             ])



    def __len__(self):
        return len(self.input_dataset)

    def __iter__(self):
        return self

    def __getitem__(self, index):

        dataA, labelA = self.input_dataset[index]
        dataB, labelB = self.output_dataset[index]

        if self.is_training:
            dataA = self.input_transform(dataA)
            dataB = self.output_transform(dataB)
            #print(dataA.size(), dataB.size())
            return dataA, dataB[0,:,:].unsqueeze(0)
        else:
            dataA = self.input_transform(dataA)
            dataB = self.output_transform(dataB)
            return dataA, dataB[0,:,:].unsqueeze(0)




class Feeder_Spectrogram_Phase(torch.utils.data.Dataset):

    def __init__(self,
                 input_folder_path,
                 output_folder_path,
                 is_training = True,
                 normalization=True,
                 mmap=False,
                 ):

        self.input_dataset = torchvision.datasets.ImageFolder(input_folder_path)
        self.output_dataset = torchvision.datasets.ImageFolder(output_folder_path)

        self.is_training = is_training



        self.input_transform = transforms.Compose([
                                             transforms.Resize(size=(384, 128), interpolation=3),
                                             transforms.ToTensor(),
                                             ])

        self.output_transform = transforms.Compose([
                                             #transforms.Resize(size=(384, 128), interpolation=3),
                                             transforms.ToTensor(),
                                             ])



    def __len__(self):
        return len(self.input_dataset)

    def __iter__(self):
        return self

    def __getitem__(self, index):

        dataA, labelA = self.input_dataset[index]
        dataB, labelB = self.output_dataset[index]

        print(index)

        if self.is_training:
            dataA = self.input_transform(dataA)
            dataA = dataA.view(9, 128, 128)

            dataB = self.output_transform(dataB)
            dataB = dataB[0:2,:,:]
            return dataA, dataB, labelA, labelB
        else:
            dataA = self.input_transform(dataA)
            dataA = dataA.view(9, 128, 128)

            dataB = self.output_transform(dataB)
            dataB = dataB[0:2,:,:]
            return dataA, dataB, labelA, labelB
