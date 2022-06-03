from matplotlib import use
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.block import ResBlk,Conv
class Generator(nn.Module):
    def __init__(self,use_bias=False,inplace=True):
        super().__init__()
        self.use_bias=use_bias
        self.inplace=inplace
        self.down1=Conv(3,32,use_bias=self.use_bias,inplace=self.inplace) #128
        self.down2=Conv(32,64,use_bias=self.use_bias,inplace=self.inplace) #64
        self.down3=Conv(64,128,use_bias=self.use_bias,inplace=self.inplace) #32
        self.down4=Conv(128,256  ,use_bias=self.use_bias,inplace=self.inplace) #16

        self.res1=nn.Sequential(
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,bias=self.use_bias),
            nn.BatchNorm2d(256,affine=True),
            nn.ReLU())

        self.res2=nn.Sequential(
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,bias=self.use_bias),
            nn.BatchNorm2d(256,affine=True),
            nn.ReLU()
        )

        self.up1_conv=Conv(256,256,use_bias=self.use_bias,inplace=self.inplace) #32
        self.up2_conv=Conv(128,128,use_bias=self.use_bias,inplace=self.inplace) #64
        self.up3_conv=Conv(64,64,use_bias=self.use_bias,inplace=self.inplace) #128
        self.up4_conv=Conv(32,32,use_bias=self.use_bias,inplace=self.inplace) #256
        
        
        self.up1=nn.Sequential(
            nn.ConvTranspose2d(256,256,kernel_size=3,stride=2,padding=1,output_padding=1,bias=self.use_bias),
            nn.BatchNorm2d(256,affine=True),
            nn.ReLU()
        )
        self.up2=nn.Sequential(
            nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1,output_padding=1,bias=self.use_bias),
            nn.BatchNorm2d(128,affine=True),
            nn.ReLU()
        )
        self.up3=nn.Sequential(
            nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,output_padding=1,bias=self.use_bias),
            nn.BatchNorm2d(64,affine=True),
            nn.ReLU()
        )
        self.up4=nn.Sequential(
            nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,padding=1,output_padding=1,bias=self.use_bias),
            nn.BatchNorm2d(32,affine=True),
            nn.ReLU()
        )
        self.last=nn.Conv2d(32,3,kernel_size=3,stride=1,padding=1,bias=self.use_bias)

    def forward(self,input):
        down1=self.down1(input)
        down2=self.down2(nn.MaxPool2d(2,2)(down1))
        down3=self.down3(nn.MaxPool2d(2,2)(down2))
        down4=self.down4(nn.MaxPool2d(2,2)(down3))
        x=self.res1(nn.MaxPool2d(2,2)(down4))
        x=self.res2(x)
        
        x=self.up1(x)
        x=self.up1_conv(down4+x)
        x=self.up2(x)
        x=self.up2_conv(down3+x)
        
        x=self.up3(x)
        x=self.up3_conv(down2+x)
        
        x=self.up4(x)
        x=self.up4_conv(down1+x)
        x=self.last(x)
        return x

class Generator2(nn.Module):
    def __init__(self,use_bias=False,inplace=True):
        super().__init__()
        self.use_bias=use_bias
        self.inplace=inplace
        self.down1=Conv(3,32,use_bias=self.use_bias,inplace=self.inplace) #128
        self.down2=Conv(32,64,use_bias=self.use_bias,inplace=self.inplace) #64
        self.down3=Conv(64,128,use_bias=self.use_bias,inplace=self.inplace) #32
        self.down4=Conv(128,256  ,use_bias=self.use_bias,inplace=self.inplace) #16

        self.res1=nn.Sequential(
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,bias=self.use_bias),
            nn.BatchNorm2d(256,affine=True),
            nn.ReLU())

        self.res2=nn.Sequential(
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,bias=self.use_bias),
            nn.BatchNorm2d(256,affine=True),
            nn.ReLU()
        )

        self.up1_conv=Conv(256,256,use_bias=self.use_bias,inplace=self.inplace) #32
        self.up2_conv=Conv(128,128,use_bias=self.use_bias,inplace=self.inplace) #64
        self.up3_conv=Conv(64,64,use_bias=self.use_bias,inplace=self.inplace) #128
        self.up4_conv=Conv(64,64,use_bias=self.use_bias,inplace=self.inplace) #256
        
        
        self.up1=nn.Sequential(
            nn.ConvTranspose2d(256,256,kernel_size=3,stride=2,padding=1,output_padding=1,bias=self.use_bias),
            nn.BatchNorm2d(256,affine=True),
            nn.ReLU()
        )
        self.up2=nn.Sequential(
            nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1,output_padding=1,bias=self.use_bias),
            nn.BatchNorm2d(128,affine=True),
            nn.ReLU()
        )
        self.up3=nn.Sequential(
            nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,output_padding=1,bias=self.use_bias),
            nn.BatchNorm2d(64,affine=True),
            nn.ReLU()
        )
        self.up4=nn.Sequential(
            nn.ConvTranspose2d(64,64,kernel_size=3,stride=2,padding=1,output_padding=1,bias=self.use_bias),
            nn.BatchNorm2d(64,affine=True),
            nn.ReLU()
        )
        self.last=nn.Conv2d(64,3,kernel_size=3,stride=1,padding=1,bias=self.use_bias)

    def forward(self,input):
        down1=self.down1(input)
        down2=self.down2(nn.MaxPool2d(2,2)(down1))
        down3=self.down3(nn.MaxPool2d(2,2)(down2))
        down4=self.down4(nn.MaxPool2d(2,2)(down3))
        x=self.res1(nn.MaxPool2d(2,2)(down4))
        x=self.res2(x)
        
        x=self.up1(x)
        x=self.up1_conv(down4+x)
        x=self.up2(x)
        x=self.up2_conv(down3+x)
        
        x=self.up3(x)
        x=self.up3_conv(down2+x)
        
        x=self.up4(x)
        x=self.up4_conv(x)
        x=self.last(x)
        return x
    
class Generator_Mod(nn.Module):
    def __init__(self,n_res=4,use_bias=False):
        super().__init__()
        
        self.n_res=n_res
        self.use_bias=use_bias
        self.down_sampling=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=1,padding=0,bias=use_bias),
            nn.BatchNorm2d(64,affine=True),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=3,bias=use_bias),
            nn.BatchNorm2d(128,affine=True),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1,bias=use_bias),
            nn.BatchNorm2d(256,affine=True),
            nn.LeakyReLU(0.2,inplace=True),
        )
        
        res_block=[]
        for i in range(n_res):
            res_block.append(ResBlk(256,256,normalize=True,use_bias=self.use_bias))
        self.res_block=nn.Sequential(*res_block)
        
        self.up_sampling = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=0, output_padding=1, bias=use_bias),
            nn.BatchNorm2d(128,affine=True),
            nn.LeakyReLU(0.2,inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0, output_padding=1, bias=use_bias),
            nn.BatchNorm2d(64,affine=True),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=0, bias=use_bias),
            nn.Tanh()
        )
    
    def forward(self,input):
        x=self.down_sampling(input)
        x=self.res_block(x)
        out=self.up_sampling(x)
        return out