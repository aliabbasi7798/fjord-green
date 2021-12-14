import numpy as np
import torch
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock
from models import FemnistCNN
from models import FjordFemnistCNN
import client


def mask_layer(param, key, p):
    mask_w = torch.ones(param.shape)
    stop_idx = int(mask_w.size(0) * p)
    prev_stop_idx = int(mask_w.size(1) * p)
    if key.find('bias') != -1:
        mask_b = torch.ones(param.shape)
        stop_idx = int(mask_b.size(0) * p)
        mask_b[stop_idx:] = 0
        return mask_b
    if key == 'conv1.weight':
        mask_w[stop_idx:] = 0
    elif key == 'conv2.weight':
        mask_w[stop_idx:, prev_stop_idx:] = 0
    elif key == 'fc1.weight':
        stop_idx = int(mask_w.size(1) * p)
        mask_w[:, stop_idx:] = 0
    return mask_w



def mask_layer1(param, p=1):
    mask = torch.ones(param.shape)
    stop_idx = int(mask.size(0) * p)
    mask[stop_idx:] = 0
    return mask


def mask_layer2(param, p=.5):
    mask = torch.ones(param.shape)
    stop_idx = int(mask.size(0) * p)
    prev_stop_idx = int(mask.size(1) * p)
    mask[stop_idx:, prev_stop_idx:] = 0
    return mask


def mask_layer3(param, p=.5):
    mask = torch.ones(param.shape)
    stop_idx = int(mask.size(1) * p)
    mask[:, stop_idx:] = 0
    return mask


if __name__ == "__main__":
    model = FjordFemnistCNN(10)
    input = torch.randn(1, 1, 28, 28)
    output = model(input, p=0.5)
    print(output)




    ''' 
    * Testing dropout
        model = FjordFemnistCNN(10)
        input = torch.randn(1,1,28,28)
        # print(input.shape)
        output = model(input, p=0.5)
        print(output)
    '''
    '''
     # cl = client.Client(k=5)
    # print(cl.max_cap)
    
    '''

    # model = ResNet(BasicBlock, [2, 2, 2, 2], )
    # input = torch.Tensor(np.ones((1, 3, 64, 64)))
    #
    # output = model.forward(input)
    # print(output.shape)

    # model = ResNet(BasicBlock, [2, 2, 2, 2], )
    # input = torch.Tensor(np.ones((1, 3, 64, 64)))
    # output = model.forward(input)
    # print(output.shape)
