from matplotlib import use
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.block import ResBlk
class CycleGanGenerator(nn.Module):
    def __init__(self,n_res=8,use_bias=False):
        super().__init__()
        
        self.n_res=n_res
        self.use_bias=use_bias
        self.down_sampling=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=1,padding=3,bias=use_bias),
            nn.InstanceNorm2d(64,affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1,bias=use_bias),
            nn.InstanceNorm2d(128,affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=3,bias=use_bias),
            nn.InstanceNorm2d(256,affine=True),
            nn.LeakyReLU(0.2),
        )
        
        res_block=[]
        for i in range(n_res):
            res_block.append(ResBlk(256,256,normalize=True,use_bias=self.use_bias))
        self.res_block=nn.Sequential(*res_block)
        
        self.up_sampling = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.InstanceNorm2d(128,affine=True),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.InstanceNorm2d(64,affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, bias=use_bias),
            nn.Tanh()
        )
    
    def forward(self,input):
        x=self.down_sampling(input)
        x=self.res_block(x)
        out=self.up_sampling(x)
        return out

