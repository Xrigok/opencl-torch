from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time
import os

class Net1(nn.Module):
    use_bn = True
    use_gp = False
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(16, 64, 3, padding=0,bias=not self.use_bn)

    def forward(self, x):
        x = self.conv1(x)

        return x

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--device',default='cpu')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = args.device
    if device.find('ocl')==0:
        if os.name == 'nt':
            torch.ops.load_library(r"build\pt_ocl.dll")
        else:
            torch.ops.load_library("build/libpt_ocl.so")
        try:
            torch.utils.rename_privateuse1_backend('ocl')
        except:
            device = device.replace('ocl','privateuseone')

    print("Using device:",device)
	
    model = Net1().to(device)
    test_data_shape=[(1,16,32,32),(1,16,64,64),(1,16,128,128)]
    with torch.no_grad():
         for i in range(2):
            for shape in test_data_shape:
                x=torch.rand(shape)
                x=x.to(device)
                model(x)
             
    
if __name__ == '__main__':
    main()
    print("Done");
