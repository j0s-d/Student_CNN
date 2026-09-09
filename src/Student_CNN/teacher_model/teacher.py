#import modules
import torch
import torch.nn as nn


 #---------- Depthwise-separable convolution block, architecture from MobileNetV1 ---------- 

class DepthSepConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthSepConv, self).__init__()

        #depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False
        )

        #normalise result
        self.bn1 = nn.BatchNorm2d(in_channels)

        #pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

        #normalise and activate results of both convolutions
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.activation(x)
        return x


#---------- Face detection network, architecture from MobileNetV1 ----------

class TeacherFaceDetector(nn.Module):
    def __init__(self):
        super(TeacherFaceDetector, self).__init__()

        #input image is 128x128, 3 channels, convolution with stride 2 reduces spatial resolution to 64x64, 16 channels
        #normal convolution is used here instead of depthwise-separable convolution, as the first layer is not computationally expensive
        #normal convolution is also used to learn relationships between the 3 input channels, which depthwise-separable convolution cannot do
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),

            #Acivation function used in MobileNetV1 is ReLU6, which is a ReLU that caps at 6.0, increasing stability
            nn.ReLU6(inplace=True)
        )

        #increases channels from 16 to 24, reduces spatial resolution from 64x64 to 32x32, as detailed in DepthSepConv class
        self.block1 = DepthSepConv(32, 64, stride=2)

        #increases channels from 24 to 32, changes spatial resolution to 16x16
        self.block2 = DepthSepConv(64, 128, stride=2)

        #increases channels from 32 to 64, keeps spatial resolution at 16x16
        #option to change to 48 channels, computationally cheaper, but reduces accuracy
        self.block3 = DepthSepConv(128, 256, stride=1)

        self.block4 = DepthSepConv(256, 256, stride=1)

        self.block5 = DepthSepConv(256, 256, stride=1)  

        self.block6 = DepthSepConv(256, 256, stride=1)  

        self.block7 = DepthSepConv(256, 256, stride=1) 


        #detection head, reduces the 64 feature channels to 5 detection channels 
        #5 detection channels correspond to [confidence score, x, y, width, height] of the detected face bounding box
        self.head = nn.Conv2d(
            256,
            5,
            kernel_size=1
        )

    def forward(self, x):
        x = self.conv1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        
        x = self.head(x)

        #function to make output easier to work with, permutes the output tensor from 
        #(batch_size, channels, height, width) to 
        # (batch_size, height, width, channels)
        x = x.permute(0, 2, 3, 1)

        return x