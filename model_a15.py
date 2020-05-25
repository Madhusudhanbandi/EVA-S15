import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torchsummary import summary


def modelup():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # Input Block
            self.convblock1 = nn.Sequential(
                nn.Conv2d(in_channels=6, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU()
              
            ) 


            # TRANSITION BLOCK 1
            self.pool = nn.MaxPool2d(2, 2)

            # CONVOLUTION BLOCK 1
            self.convblock2 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU()
              
            )

            self.convblock3 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU()
              
            )

            self.convblock4 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()
              
            )


            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

            self.up_conv1 = nn.Sequential(
                nn.Conv2d(in_channels=768, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()

            )

            self.up_conv11 = nn.Sequential(
                nn.Conv2d(in_channels=768, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU()

            )

            self.up_conv2 = nn.Sequential(
                nn.Conv2d(in_channels=640, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )

            self.up_conv21 = nn.Sequential(
                nn.Conv2d(in_channels=640, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )

            self.up_conv3 = nn.Sequential(
                nn.Conv2d(in_channels=192, out_channels=1, kernel_size=(3, 3), padding=1, bias=False),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

            self.up_conv31 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(3, 3), padding=1, bias=False),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

            
         

            self.up_conv4 = nn.Sequential(
                nn.Conv2d(in_channels=192, out_channels=1, kernel_size=(3, 3), padding=1, bias=False),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

            self.up_conv41 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(3, 3), padding=1, bias=False),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

                    


        def forward(self, x):

            x2 = self.pool(self.convblock1(x)) #112 c=64

            x3 = self.pool(self.convblock2(x2)) #56 c=128

            x4 = self.pool(self.convblock3(x3)) #28 c=256
            
            x5 = self.pool(self.convblock4(x4)) #14 c=512

            x6 = self.upsample(x5) #28 c=512

            x7 = torch.cat([x6,x4],dim=1) 

            x8 = self.upsample(self.up_conv1(x7)) # c=512+256  o=512  s=56

            x81 = self.upsample(self.up_conv11(x7))


            x9 = torch.cat([x8,x3],dim=1) 

            x91 = torch.cat([x81,x3],dim=1) 

            x10 = self.upsample(self.up_conv2(x9))

            x101 = self.upsample(self.up_conv21(x91))

            x11 = torch.cat([x10,x2],dim=1) 

            x111 = torch.cat([x101,x2],dim=1) 

            x12 = self.upsample(self.up_conv3(x11))

            x13 = self.up_conv4(x111)
           
            

            return x12,x13
        
    return Net