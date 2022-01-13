#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""The code below is taken from
https://github.com/JunjH/Revisiting_Single_Depth_Estimation
Revisiting Single Image Depth Estimation: Toward Higher Resolution Maps With Accurate Object Boundaries
Junjie Hu and Mete Ozay and Yan Zhang and Takayuki Okatani
WACV 2019

ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""


import math

import numpy as np
import torch
import torch.nn.parallel
from PIL import Image
from torch import nn as nn
from torch.nn import functional as F
from torch.utils import model_zoo as model_zoo
from torchvision import transforms

accimage = None


__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    r"""Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]))
    return model


def resnet34(pretrained=False, **kwargs):
    r"""Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet34"]))
    return model


def resnet50(pretrained=False, **kwargs):
    r"""Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(
            model_zoo.load_url(
                model_urls["resnet50"], "pretrained_model/encoder"
            )
        )
    return model


def resnet101(pretrained=False, **kwargs):
    r"""Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet101"]))
    return model


def resnet152(pretrained=False, **kwargs):
    r"""Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet152"]))
    return model


class model(nn.Module):
    def __init__(self, Encoder, num_features, block_channel):

        super(model, self).__init__()

        self.E = Encoder
        self.D = D(num_features)
        self.MFF = MFF(block_channel)
        self.R = R(block_channel)

    def forward(self, x):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(
            x_block1,
            x_block2,
            x_block3,
            x_block4,
            [x_decoder.size(2), x_decoder.size(3)],
        )
        out = self.R(torch.cat((x_decoder, x_mff), 1))

        return out


class _UpProjection(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_UpProjection, self).__init__()

        self.conv1 = nn.Conv2d(
            num_input_features,
            num_output_features,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(
            num_output_features,
            num_output_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1_2 = nn.BatchNorm2d(num_output_features)

        self.conv2 = nn.Conv2d(
            num_input_features,
            num_output_features,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(num_output_features)

    def forward(self, x, size):
        x = F.upsample(x, size=size, mode="bilinear")
        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))

        out = self.relu(bran1 + bran2)

        return out


class E_resnet(nn.Module):
    def __init__(self, original_model, num_features=2048):
        super(E_resnet, self).__init__()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool

        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)

        return x_block1, x_block2, x_block3, x_block4


class D(nn.Module):
    def __init__(self, num_features=2048):
        super(D, self).__init__()
        self.conv = nn.Conv2d(
            num_features,
            num_features // 2,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        num_features = num_features // 2
        self.bn = nn.BatchNorm2d(num_features)

        self.up1 = _UpProjection(
            num_input_features=num_features,
            num_output_features=num_features // 2,
        )
        num_features = num_features // 2

        self.up2 = _UpProjection(
            num_input_features=num_features,
            num_output_features=num_features // 2,
        )
        num_features = num_features // 2

        self.up3 = _UpProjection(
            num_input_features=num_features,
            num_output_features=num_features // 2,
        )
        num_features = num_features // 2

        self.up4 = _UpProjection(
            num_input_features=num_features,
            num_output_features=num_features // 2,
        )
        num_features = num_features // 2

    def forward(self, x_block1, x_block2, x_block3, x_block4):
        x_d0 = F.relu(self.bn(self.conv(x_block4)))
        x_d1 = self.up1(x_d0, [x_block3.size(2), x_block3.size(3)])
        x_d2 = self.up2(x_d1, [x_block2.size(2), x_block2.size(3)])
        x_d3 = self.up3(x_d2, [x_block1.size(2), x_block1.size(3)])
        x_d4 = self.up4(x_d3, [x_block1.size(2) * 2, x_block1.size(3) * 2])

        return x_d4


class MFF(nn.Module):
    def __init__(self, block_channel, num_features=64):

        super(MFF, self).__init__()

        self.up1 = _UpProjection(
            num_input_features=block_channel[0], num_output_features=16
        )

        self.up2 = _UpProjection(
            num_input_features=block_channel[1], num_output_features=16
        )

        self.up3 = _UpProjection(
            num_input_features=block_channel[2], num_output_features=16
        )

        self.up4 = _UpProjection(
            num_input_features=block_channel[3], num_output_features=16
        )

        self.conv = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x_block1, x_block2, x_block3, x_block4, size):
        x_m1 = self.up1(x_block1, size)
        x_m2 = self.up2(x_block2, size)
        x_m3 = self.up3(x_block3, size)
        x_m4 = self.up4(x_block4, size)

        x = self.bn(self.conv(torch.cat((x_m1, x_m2, x_m3, x_m4), 1)))
        x = F.relu(x)

        return x


class R(nn.Module):
    def __init__(self, block_channel):

        super(R, self).__init__()

        num_features = 64 + block_channel[3] // 32
        self.conv0 = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False,
        )
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features)

        self.conv2 = nn.Conv2d(
            num_features, 1, kernel_size=5, stride=1, padding=2, bias=True
        )

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)

        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        x2 = self.conv2(x1)

        return x2


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class Scale:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = self.changeScale(image, self.size)

        return image

    def changeScale(self, img, size, interpolation=Image.BILINEAR):
        ow, oh = size

        return img.resize((ow, oh), interpolation)


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = self.centerCrop(image, self.size)

        return image

    def centerCrop(self, image, size):
        w1, h1 = image.size
        tw, th = size

        if w1 == tw and h1 == th:
            return image

        x1 = int(round((w1 - tw) / 2.0))
        y1 = int(round((h1 - th) / 2.0))

        image = image.crop((x1, y1, tw + x1, th + y1))

        return image


class ToTensor:
    r"""Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, image):
        image = self.to_tensor(image)

        return image

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                "pic should be PIL Image or ndarray. Got {}".format(type(pic))
            )

        if isinstance(pic, np.ndarray):

            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float().div(255)

        if accimage is not None and isinstance(pic, accimage.Image):  # type: ignore
            nppic = np.zeros(  # type: ignore[unreachable]
                [pic.channels, pic.height, pic.width], dtype=np.float32
            )
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == "I":
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == "I;16":
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes())  # type: ignore
            )
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == "YCbCr":
            nchannel = 3
        elif pic.mode == "I;16":
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = self.normalize(image, self.mean, self.std)

        return image

    def normalize(self, tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)

        return tensor


def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet50(pretrained=False)
        Encoder = E_resnet(original_model)
        model1 = model(
            Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048]
        )
    if is_densenet:
        # original_model = dendensenet161(pretrained=False)
        # Encoder = E_densenet(original_model)
        # model1 = model(
        #    Encoder, num_features=2208, block_channel=[192, 384, 1056, 2208]
        # )
        raise NotImplementedError()
    if is_senet:
        # original_model = senet154(pretrained=False)
        # Encoder = E_senet(original_model)
        # model1 = model(
        #    Encoder, num_features=2048, block_channel=[256, 512, 1024, 2048]
        # )
        raise NotImplementedError()
    return model1


class MonoDepthEstimator:
    def __init__(self, checkpoint="./pretrained_model/model_resnet"):
        self.model = define_model(
            is_resnet=True, is_densenet=False, is_senet=False
        )
        self.model = torch.nn.DataParallel(self.model).cuda()
        cpt = torch.load(checkpoint)
        if "state_dict" in cpt:
            cpt = cpt["state_dict"]
        self.model.load_state_dict(cpt)
        self.model.eval()
        self.init_preprocessor()

    def init_preprocessor(self):
        __imagenet_stats = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }

        self.transform = transforms.Compose(
            [
                Scale([320, 240]),
                # CenterCrop([304, 228]),
                ToTensor(),
                Normalize(__imagenet_stats["mean"], __imagenet_stats["std"]),
            ]
        )

    def preprocess(self, image):
        image_torch = self.transform(image).unsqueeze(0)
        return image_torch.cuda()

    def compute_depth(self, image):
        # Input: image is a PIL image
        # Output: depth is a numpy array
        image_torch = self.preprocess(image)
        # print(image_torch.size())
        depth_torch = self.model(image_torch)
        depth = (
            depth_torch.view(depth_torch.size(2), depth_torch.size(3))
            .data.cpu()
            .numpy()
        )
        return depth
