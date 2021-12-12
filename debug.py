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
    # to = torch.ones([20,10,5,5])
    # mask = mask_layer2(to,p=0.5)
    # print(mask)

    # t1 = torch.ones([10,1,5,5])
    # t2 = torch.ones([10,1,5,5])
    # t3 = torch.ones([10,1,5,5])
    # t4 = torch.ones([10,1,5,5])
    #
    # t_list = [t1,t2,t3,t4]
    #
    # mask1 = mask_layer(0.2)
    # mask2 = mask_layer(0.4)
    # mask3 = mask_layer(0.5)
    # mask4 = mask_layer(1)
    #
    # t_mask = [mask1,mask2,mask3,mask4]
    #
    # smask = mask1+mask4+mask2+mask3
    #
    # t_total = torch.zeros([10,1,5,5])
    # for x,m in zip(t_list,t_mask):
    #     t_total += x.masked_fill(m==0,0)
    #
    # # t_total = t1.masked_fill(mask1==0,0)+t2.masked_fill(mask2==0,0)+t3.masked_fill(mask3==0,0)+t4.masked_fill(mask4==0,0)
    # # t_total = t_total
    # print(t_total)

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
