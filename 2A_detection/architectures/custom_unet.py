import torch
from torch import nn


class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding):
        super(DSConv, self).__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class FastVGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(FastVGGBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = DSConv(in_channels, middle_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = DSConv(middle_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class FastNestedUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False):
        super(FastNestedUNet, self).__init__()

        # nb_filter = [32, 64, 128, 256, 512]
        nb_filter = [64, 128, 256, 512, 1024]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = FastVGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = FastVGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = FastVGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = FastVGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = FastVGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = FastVGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = FastVGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = FastVGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = FastVGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = FastVGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = FastVGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = FastVGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = FastVGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = FastVGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = FastVGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = DSConv(nb_filter[0], num_classes, kernel_size=1, padding=0)
            self.final2 = DSConv(nb_filter[0], num_classes, kernel_size=1, padding=0)
            self.final3 = DSConv(nb_filter[0], num_classes, kernel_size=1, padding=0)
            self.final4 = DSConv(nb_filter[0], num_classes, kernel_size=1, padding=0)
        else:
            self.final = DSConv(nb_filter[0], num_classes, kernel_size=1, padding=0)

        self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            output = self.sigmoid(output)
            return output
