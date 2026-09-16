#import modules
import torch
import torch.nn as nn
import torch.nn.functional as F

#depthwise convolution block
def depthwise_conv(ch_in, stride=1):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, stride=stride, groups=ch_in, bias=False),
            nn.BatchNorm2d(ch_in),
            nn.ReLU6(inplace=True),
        )
    )

#pointwise convolution used to create depthwise-separable sequence
def conv1x1(ch_in, ch_out):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )

#3x3 convolution block
def conv3x3(ch_in, ch_out, stride):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )

#---------- Inverted block class from MobileNetV2 ---------

class InvertedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, expand_ratio, stride):
        super(InvertedBlock, self).__init__()

        self.stride = stride
        assert stride in [1,2]

        hidden_dim = ch_in * expand_ratio

        self.use_res_connect = self.stride==1 and ch_in==ch_out

        layers = []
        if expand_ratio != 1:
            layers.append(conv1x1(ch_in, hidden_dim))
        layers.extend([
            #dw
            depthwise_conv(hidden_dim, stride=stride),
            #pw
            conv1x1(hidden_dim, ch_out)
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)

#--------- Modified MobileNetV2, same as teacher but fpn network uses 64 channels- 1,338,815 trainable parameters---------

class StudentFaceDetector(nn.Module):
    def __init__(self, ch_in=3, fpn_channels = 64):
        super(StudentFaceDetector, self).__init__()

        #initial normal convolution, since first layer is not computationally expensive (only 3 channels)
        self.stem_conv = conv3x3(ch_in, 32, stride=2)


        #using multiple sequential stages instead of a one sequential
        #this allows multiple detection heads to be used at different spatial resolutions
        #uses 12 convolution blocks
        self.stage1 = nn.Sequential(
            InvertedBlock(32, 16, 1, 1),
        )

        self.stage2 = nn.Sequential(
            InvertedBlock(16, 24, 6, 2),
            InvertedBlock(24, 24, 6, 1)
        )

        self.stage3 = nn.Sequential(
            InvertedBlock(24, 32, 6, 2),
            InvertedBlock(32, 32, 6, 1)
        )

        self.stage4 = nn.Sequential(
            InvertedBlock(32, 64, 6, 2),
            InvertedBlock(64, 64, 6, 1)
        )

        self.stage5 = nn.Sequential(
            InvertedBlock(64, 96, 6, 1),
            InvertedBlock(96, 96, 6, 1)
        )

        self.stage6 = nn.Sequential(
            InvertedBlock(96, 160, 6, 2),
            InvertedBlock(160, 160, 6, 1)
        )

        self.stage7 = nn.Sequential(
            InvertedBlock(160, 320, 6, 1)
        )

        #lateral projections
        #used to project small, medium and large predictions to common feature dimension
        self.lat_small = nn.Conv2d(32, fpn_channels, kernel_size=1)
        self.lat_medium = nn.Conv2d(64, fpn_channels, kernel_size=1)
        self.lat_large = nn.Conv2d(320, fpn_channels, kernel_size=1)

        #feature pyramid network
        #3x3 convolutions after feature fusion
        self.fpn_small = conv3x3(fpn_channels, fpn_channels, stride=1)
        self.fpn_medium = conv3x3(fpn_channels, fpn_channels, stride=1)
        self.fpn_large = conv3x3(fpn_channels, fpn_channels, stride=1)


        #detection heads for different sized faces after feature pyramid network
        #outputs 5 detection channels [confidence, x, y, width, height]
        self.small_face = nn.Conv2d(fpn_channels, 5, kernel_size=1)
        self.medium_face = nn.Conv2d(fpn_channels, 5, kernel_size=1)
        self.large_face = nn.Conv2d(fpn_channels, 5, kernel_size=1)

    #forward pass over model
    def forward(self, x):
        x = self.stem_conv(x)

        x = self.stage1(x)
        x = self.stage2(x)

        #small faces
        xs = self.stage3(x)

        #medium faces
        xm = self.stage4(xs)

        x = self.stage5(xm)
        x = self.stage6(x)

        #large faces
        xl = self.stage7(x)

        #project predictions to 64 feature channels
        ps = self.lat_small(xs)
        pm = self.lat_medium(xm)
        pl = self.lat_large(xl)

        #feature pyramid network
        #idea is to upsample tensor with smaller spatial resolution (eg pl)
        #and add to tensor with greater spatial resolution, passing on semantic information
        #the combined tensor is then passed to a 3x3 convolution
        pl = self.fpn_large(pl)

        pm = pm + F.interpolate(
            pl,
            size = pm.shape[-2:],
            mode = 'nearest'
        )
        pm = self.fpn_medium(pm)

        ps = ps + F.interpolate(
            pm,
            size = ps.shape[-2:],
            mode = 'nearest'
        )
        ps = self.fpn_small(ps)

        #detection heads
        small_faces = self.small_face(ps)
        medium_faces = self.medium_face(pm)
        large_faces = self.large_face(pl)

        #converts [B, D, H, W] to [B, H, W, D]
        small_faces = small_faces.permute(0, 2, 3, 1)
        medium_faces = medium_faces.permute(0, 2, 3, 1)
        large_faces = large_faces.permute(0, 2, 3, 1)

        #reshapes 2D grid of predictions into 1D list of predictions
        small_faces = small_faces.reshape(x.size(0), -1, 5)
        medium_faces = medium_faces.reshape(x.size(0), -1, 5)
        large_faces = large_faces.reshape(x.size(0), -1, 5)

        #concatenates predictions into a list of small-grid followed by medium-grid then large-grid 
        return torch.cat([small_faces, medium_faces, large_faces], dim=1)


if __name__ == '__main__':
    model = StudentFaceDetector()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total:     {total:,}")
    print(f"Trainable: {trainable:,}")