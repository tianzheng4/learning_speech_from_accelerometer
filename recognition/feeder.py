# sys
import pickle

# torch
import torch
# from torch.autograd import Variable
import torchvision
from torchvision import transforms
import numpy as np

from PIL import Image
import glob


class Feeder(torch.utils.data.Dataset):

    def __init__(self,
                 folder_path,
                 is_training = True,
                 normalization=True,
                 mmap=False,
                 target_index=None
                 ):

        self.dataset = torchvision.datasets.ImageFolder(folder_path)

        if target_index != None:
            idx = []
            for i in range(len(self.dataset)):
                if self.dataset[i][1] == target_index:
                    idx.append(i)

            self.dataset = [self.dataset[i] for i in iter(idx)]

        self.is_training = is_training

        if self.dataset == None:
            print('no such folder')
            exit()


        self.train_transform = transforms.Compose([
                                             # transforms.Pad(padding=8),
                                             # transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0), interpolation=3),
                                             transforms.Resize(size=(224, 224), interpolation=3),
                                             #torchvision.transforms.RandomRotation(60),
                                             #transforms.Resize(size=(224, 224)),
                                             transforms.ToTensor(),
                                             ])

        self.test_transform = transforms.Compose([#transforms.CenterCrop((128, 32)),
                                             transforms.Resize(size=(224, 224), interpolation=3),
                                             transforms.ToTensor(),
                                             ])


    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return self

    def __getitem__(self, index):

        data, label = self.dataset[index]

        if self.is_training:
            data = self.train_transform(data)
            return data, label
        else:
            data = self.test_transform(data)
            return data, label


class MultiDatasetFeeder(torch.utils.data.Dataset):

    def __init__(self,
                 folder_paths,
                 is_training = True,
                 normalization=True,
                 mmap=False,
                 target_index=None
                 ):

        ## combine datasets
        self.dataset = []
        for path in iter(folder_paths):
            dataset = torchvision.datasets.ImageFolder(path)
            for i in range(len(dataset)):
                self.dataset.append(dataset[i])

        ## select data from a specific class (index)
        if target_index != None:
            idx = []
            for i in range(len(self.dataset)):
                if self.dataset[i][1] == target_index:
                    idx.append(i)

            self.dataset = [self.dataset[i] for i in iter(idx)]


        self.is_training = is_training

        if self.dataset == None:
            print('no such folder')
            exit()


        self.train_transform = transforms.Compose([
                                             # transforms.Pad(padding=8),
                                             # transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0), interpolation=3),
                                             transforms.Resize(size=(224, 224), interpolation=3),
                                             #torchvision.transforms.RandomRotation(60),
                                             #transforms.Resize(size=(224, 224)),
                                             transforms.ToTensor(),
                                             ])

        self.test_transform = transforms.Compose([#transforms.CenterCrop((128, 32)),
                                             transforms.Resize(size=(224, 224), interpolation=3),
                                             transforms.ToTensor(),
                                             ])


    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return self

    def __getitem__(self, index):

        data, label = self.dataset[index]

        if self.is_training:
            data = self.train_transform(data)
            return data, label
        else:
            data = self.test_transform(data)
            return data, label



class ImageFeeder:

    def __init__(self, label=None):
        self.transform = transforms.Compose([#transforms.CenterCrop((128, 32)),
                                             transforms.Resize(size=(224, 224), interpolation=3),
                                             transforms.ToTensor(),
                                             ])
        self.label = label

    def __call__(self, *img_paths):

        img_tensor = []
        for path in img_paths:
            img = Image.open(path)
            img_tensor.append(self.transform(img).unsqueeze(0))



        if self.label != None:
            label = torch.zeros([len(img_paths), ], dtype=torch.int64) + self.label
            return torch.cat(img_tensor), label

        return torch.cat(img_tensor)








def test(folder_path='./dataset/'):
    import matplotlib.pyplot as plt

    loader = torch.utils.data.DataLoader(
        dataset=Feeder(folder_path),
        batch_size=64,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    for data_batch, label_batch in loader:
        print(data_batch.type, label_batch.type)
        print(data_batch.size(), data_batch[0].max(), data_batch[0].min())
        print(label_batch)



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    test('./dataset/Image/')
