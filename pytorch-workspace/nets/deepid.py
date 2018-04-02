import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def DeepID_256_gray(**kwargs):
    model = deepid_model_256_gray(**kwargs)
    return model

def DeepID_128_gray(**kwargs):
    model = deepid_model_128_gray(**kwargs)
    return model

def DeepID_256(**kwargs):
    model = deepid_model_256(**kwargs)
    return model

class deepid_model_256_gray(nn.Module):
    def __init__(self, num_classes=10572, input_size=(1,256,256)):
        super(deepid_model_256_gray, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )
        self.flat_feature = self.get_flat_feature(input_size, self.features)
        #self.fc_embedding = nn.Linear(self.flat_feature, 256)
        self.fc_embedding = nn.Linear(27648, 256)
        self.fc_classifier= nn.Linear(256, num_classes)
     
    def forward(self, x):
        x = self.features(x)
        x = x.view(self.flat_feature, -1)
        x = self.fc_embedding(x)
        logits = self.fc_classifier(x)
        return logits, x

    def get_flat_feature(self, in_size, fts):
        f = fts(Variable(torch.ones(1,*in_size)))
        print(f.size())
        return int(np.prod(f.size()[1:]))
    
##########################################

class deepid_model_128_gray(nn.Module):
    def __init__(self, num_classes=10572, input_size=(1,128,128)):
        super(deepid_model_128_gray, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
        )
        self.flat_feature = self.get_flat_feature(input_size, self.features)
        #self.fc_embedding = nn.Linear(self.flat_feature, 256)
        self.fc_embedding = nn.Linear(6272, 256)
        self.fc_classifier= nn.Linear(256, num_classes)
     
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        #x = x.view(self.flat_feature, -1)
        x = self.fc_embedding(x)
        
        logits = self.fc_classifier(x)
        return logits, x

    def get_flat_feature(self, in_size, fts):
        f = fts(Variable(torch.ones(1,*in_size)))
        print(f.size())
        return int(np.prod(f.size()[1:]))
    
##########################################



class deepid_model_256(nn.Module):
    def __init__(self, num_classes=10572):
        super(deepid_model_256, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=6, stride=2, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )
        self.fc_embedding = nn.Linear(1321, 256)
        self.fc_classifier= nn.Linear(256, num_classes)
     
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_embedding(x)
        x = self.fc_classifier(x)
        return x