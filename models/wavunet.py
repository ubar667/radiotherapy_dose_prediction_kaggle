import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *
from pytorch_wavelets import IDWT

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model
class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers=18, pretrained=False, num_input_images=12):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        #x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features

class DepthWaveProgressiveDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthWaveProgressiveDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])  # the last layer is removed since it is the full scale

        self.J = 1
        self.inverse_wt = IDWT(wave="haar", mode="zero")

        # decoder
        self.convs = OrderedDict()
        for i in range(4, 0, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]

            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out, use_refl=True)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out, use_refl=True)

            # [0, 1, 2, 3] as scale options, will use scales [1, 2, 3, 4] to compute wavelet coefficients
            if i == 4:
                # LL
                self.convs[("waveconv", i, 0)] = nn.Sequential(*[Conv1x1(self.num_ch_dec[i], self.num_ch_dec[i] // 4),
                                                                 nn.LeakyReLU(0.1, inplace=True),
                                                                 Conv3x3(self.num_ch_dec[i] // 4, 1,
                                                                         use_refl=True)])  # low frequency

            self.convs[("waveconv", i, 1)] = nn.Sequential(*[Conv1x1(self.num_ch_dec[i], self.num_ch_dec[i]),
                                                             nn.LeakyReLU(0.1, inplace=True),
                                                             Conv3x3(self.num_ch_dec[i], 3,
                                                                     use_refl=True)])

            # split between positive and negative parts
            self.convs[("waveconv", i, -1)] = nn.Sequential(*[Conv1x1(self.num_ch_dec[i], self.num_ch_dec[i]),
                                                              nn.LeakyReLU(0.1, inplace=True),
                                                              Conv3x3(self.num_ch_dec[i], 3,
                                                                      use_refl=True)])

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.ReLU()
        self.tanh = nn.ReLU()

    def get_coefficients(self, input_features, scale=1, return_ll=False):
        """
        Takes features maps at scale s as input and returns tuple (LL, [LH, HL, HH])
        """

        yl = None
        if return_ll:
            yl = 2**scale * self.sigmoid(self.convs[("waveconv", scale, 0)](input_features))
        yh = 2**(scale-1) * self.sigmoid(self.convs[("waveconv", scale, 1)](input_features)).unsqueeze(1) - \
             2**(scale-1) * self.sigmoid(self.convs[("waveconv", scale, -1)](input_features)).unsqueeze(1)
        return yl, yh

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]

        for i in range(4, 0, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i == 4:
                # LL, (LH, HL, HH)
                yl, yh = self.get_coefficients(x, scale=i, return_ll=True)
            else:
                # compute only high coefficients (LH, HL, HH) and keep previous low LL
                _, yh = self.get_coefficients(x, scale=i, return_ll=False)

            # log coefficients
            self.outputs[("wavelets", i - 1, "LL")] = yl
            self.outputs[("wavelets", i - 1, "LH")] = yh[:, :, 0]
            self.outputs[("wavelets", i - 1, "HL")] = yh[:, :, 1]
            self.outputs[("wavelets", i - 1, "HH")] = yh[:, :, 2]

            yl = self.inverse_wt((yl, list([yh])))

            self.outputs[("disp", i - 1)] = torch.clamp(yl / 2**(i-1), 0, 1)

        return self.outputs

    
class DenseModel(nn.Module):
    def __init__(self,  num_layers, output_scales, device="cuda"):
        super(DenseModel, self).__init__()
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        pretrained = False
        self.encoder = resnet_multiimage_input(num_layers, pretrained, 12)
      

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4
            
        num_output_channels=1
        self.num_output_channels = num_output_channels
        self.use_skips = True
        self.upsample_mode = 'nearest'
        self.scales = output_scales

        self.num_ch_enc = self.num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])  # the last layer is removed since it is the full scale

        self.J = 1
        self.inverse_wt = IDWT(wave="haar", mode="zero")

        # decoder
        self.convs = OrderedDict()
        for i in range(4, 0, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]

            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out, use_refl=True)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out, use_refl=True)

            # [0, 1, 2, 3] as scale options, will use scales [1, 2, 3, 4] to compute wavelet coefficients
            if i == 4:
                # LL
                self.convs[("waveconv", i, 0)] = nn.Sequential(*[Conv1x1(self.num_ch_dec[i], self.num_ch_dec[i] // 4),
                                                                 nn.LeakyReLU(0.1, inplace=True),
                                                                 Conv3x3(self.num_ch_dec[i] // 4, 1,
                                                                         use_refl=True)])  # low frequency

            self.convs[("waveconv", i, 1)] = nn.Sequential(*[Conv1x1(self.num_ch_dec[i], self.num_ch_dec[i]),
                                                             nn.LeakyReLU(0.1, inplace=True),
                                                             Conv3x3(self.num_ch_dec[i], 3,
                                                                     use_refl=True)])

            # split between positive and negative parts
            self.convs[("waveconv", i, -1)] = nn.Sequential(*[Conv1x1(self.num_ch_dec[i], self.num_ch_dec[i]),
                                                              nn.LeakyReLU(0.1, inplace=True),
                                                              Conv3x3(self.num_ch_dec[i], 3,
                                                                      use_refl=True)])

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.ReLU()
        self.tanh = nn.ReLU()

    def get_coefficients(self, input_features, scale=1, return_ll=False):
        """
        Takes features maps at scale s as input and returns tuple (LL, [LH, HL, HH])
        """

        yl = None
        if return_ll:
            yl = 2**scale * self.sigmoid(self.convs[("waveconv", scale, 0)](input_features))
        yh = 2**(scale-1) * self.sigmoid(self.convs[("waveconv", scale, 1)](input_features)).unsqueeze(1) - \
             2**(scale-1) * self.sigmoid(self.convs[("waveconv", scale, -1)](input_features)).unsqueeze(1)
        return yl, yh


    def forward(self, input_image):
        x = input_image[0]
        mask = input_image[1]
        self.features = []
        #x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        self.outputs = {}
        input_features = self.features
        # decoder
        x = input_features[-1]

        for i in range(4, 0, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i == 4:
                # LL, (LH, HL, HH)
                yl, yh = self.get_coefficients(x, scale=i, return_ll=True)
            else:
                # compute only high coefficients (LH, HL, HH) and keep previous low LL
                _, yh = self.get_coefficients(x, scale=i, return_ll=False)

            # log coefficients
            self.outputs[("wavelets", i - 1, "LL")] = yl
            self.outputs[("wavelets", i - 1, "LH")] = yh[:, :, 0]
            self.outputs[("wavelets", i - 1, "HL")] = yh[:, :, 1]
            self.outputs[("wavelets", i - 1, "HH")] = yh[:, :, 2]

            yl = self.inverse_wt((yl, list([yh])))

            self.outputs[("disp", i - 1)] = self.tanh(yl)

        return self.outputs
