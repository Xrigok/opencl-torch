import torch
from torch import nn
import torch.nn.functional as F
torch.ops.load_library("build/libpt_ocl.so")

class Net(nn.Module):
    use_bn = True
    use_gp = False
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=0,bias=not self.use_bn)

    def forward(self, x):
        x = self.conv1(x)
        return x

device="privateuseone:0"
a = torch.randn(1,3,56,56).to(device)

#net=Net().to(device)
#net.eval()
#net(a)
