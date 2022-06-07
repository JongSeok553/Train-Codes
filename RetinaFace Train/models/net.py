import time
from typing import Sequence
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )
    
# def Identity(input):
#   return nn.Identity(input)

# def hardswish(input):
#     return nn.Hardswish(input)
def ConvBNIdentity(input, output, kernel_size, stride, padding=0, groups=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(input, output, kernel_size, stride, padding, groups=groups, bias=bias),
        nn.BatchNorm2d(output, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
        nn.Identity()
    )

def ConvBNHardswish(input, output, kernel_size, stride, padding=0, groups=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(input, output, kernel_size, stride, padding, groups=groups, bias=bias),
        nn.BatchNorm2d(output, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
        nn.Hardswish()
    )

def ConvBNReLU(input, output, kernel_size, stride, padding=0, groups=1, bias=False):
    return nn.Sequential(
        nn.Conv2d(input, output, kernel_size, stride, padding, groups=groups, bias=bias),
        nn.BatchNorm2d(output, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True)
    )

def SqueezeExcitation(input, output, kernel_size, stride):
    return nn.Sequential(
        nn.Conv2d(input, output, kernel_size, stride),
        nn.ReLU(inplace=True),
        nn.Conv2d(output, input, kernel_size, stride)
    )

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out

class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out



class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky = 0.1),    # 3
            conv_dw(8, 16, 1),   # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1), # 59 + 32 = 91
            conv_dw(128, 128, 1), # 91 + 32 = 123
            conv_dw(128, 128, 1), # 123 + 32 = 155
            conv_dw(128, 128, 1), # 155 + 32 = 187
            conv_dw(128, 128, 1), # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2), # 219 +3 2 = 241
            conv_dw(256, 256, 1), # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


class FPN_V3(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN_V3,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1

        self.output1  = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2  = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3  = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1  = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2  = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge3  = conv_bn(out_channels, out_channels, leaky = leaky)
       

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1  = self.output1(input[0])
        output2  = self.output2(input[1])
        output3  = self.output3(input[2])
       
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out

class MobileNetV3(nn.Module):
    def __init__(self):
        super(MobileNetV3, self).__init__()
        self.InvertedResidual_1 = nn.Sequential(
            ConvBNHardswish(3, 16, 3, 2, 1),
            ConvBNReLU(16, 16, 3, 1, 1, 16),
            ConvBNIdentity(16, 16, 1, 1),
            ConvBNReLU(16, 64, 1, 1),

            ConvBNReLU(64, 64, 3, 2, 1, 64),
            ConvBNIdentity(64, 24, 1, 1),
            ConvBNReLU(24, 72, 1, 1),

            ConvBNReLU(72, 72, 3, 1, 1, 72),
            ConvBNIdentity(72, 24, 1, 1),
            ConvBNReLU(24, 72, 1, 1),

            ConvBNReLU(72, 72, 5, 2, 2, 72),
            SqueezeExcitation(72, 24, 1, 1),
            ConvBNIdentity(72, 40, 1, 1),
        )

        self.InvertedResidual_2 = nn.Sequential(
            ConvBNReLU(40, 120, 1, 1),
            ConvBNReLU(120, 120, 5, 1, 2, 120),
            SqueezeExcitation(120, 32, 1, 1),
            ConvBNIdentity(120, 40, 1, 1),

            ConvBNReLU(40, 120, 1, 1),
            ConvBNReLU(120, 120, 5, 1, 2, 120),
            SqueezeExcitation(120, 32, 1, 1),
            ConvBNIdentity(120, 40, 1, 1),

            ConvBNHardswish(40, 240, 1, 1),
            ConvBNHardswish(240, 240, 3, 2, 1, 240),
            ConvBNIdentity(240, 80, 1, 1),

            ConvBNHardswish(80, 200, 1, 1),
            ConvBNHardswish(200, 200, 3, 1, 1, 200),
            ConvBNIdentity(200, 80, 1, 1),

            ConvBNHardswish(80, 184, 1, 1),
            ConvBNHardswish(184, 184, 3, 1, 1, 184),
            ConvBNIdentity(184, 80,1, 1),

            ConvBNHardswish(80, 184, 1, 1),
            ConvBNHardswish(184, 184, 3, 1, 1, 184),
            ConvBNIdentity(184, 80, 1, 1)
        )

        self.InvertedResidual_3 = nn.Sequential(
            ConvBNHardswish(80, 480, 1, 1),
            ConvBNHardswish(480, 480, 3, 1, 1, 480),
            SqueezeExcitation(480, 120, 1, 1),
            ConvBNIdentity(480, 112, 1, 1),

            ConvBNHardswish(112, 672, 1, 1),
            ConvBNHardswish(672, 672, 3, 1, 1, 672),
            SqueezeExcitation(672, 168, 1, 1),
            ConvBNIdentity(672, 112, 1, 1),

            ConvBNHardswish(112, 672, 1, 1),
            ConvBNHardswish(672, 672, 5, 2, 2, 672),
            SqueezeExcitation(672, 168, 1, 1),
            ConvBNIdentity(672, 160, 1, 1),

            ConvBNHardswish(160, 960, 1, 1),
            ConvBNHardswish(960, 960, 5, 1, 2, 960),
            SqueezeExcitation(960, 240, 1, 1),
            ConvBNIdentity(960, 160, 1, 1),

            ConvBNHardswish(160, 960, 1, 1)
        )
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1)
        )
        self.Classifier = nn.Sequential(
            nn.Linear(960, 1280, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, 1000, bias=True)
        )
    def forward(self, x):
        # x = self.stage1(x)
        x = self.InvertedResidual_1(x)
        x = self.InvertedResidual_2(x)
        x = self.InvertedResidual_3(x)
       
        x = self.avgpool(x)
        x = x.view(-1, 960)
        x = self.Classifier(x)
        return x
