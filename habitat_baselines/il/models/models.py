import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultitaskCNN(nn.Module):
    def __init__(
        self,
        num_classes=41,
        only_encoder=False,
        pretrained=True,
        checkpoint_path="data/eqa/eqa_cnn_pretrain/checkpoints/epoch_5.ckpt",
        freeze_encoder=False,
    ):
        super(MultitaskCNN, self).__init__()

        self.num_classes = num_classes
        self.only_encoder = only_encoder

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 8, 5),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(8, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(16, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(32, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(32, 512, 5),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )

        self.encoder_seg = nn.Conv2d(512, self.num_classes, 1)
        self.encoder_depth = nn.Conv2d(512, 1, 1)
        self.encoder_ae = nn.Conv2d(512, 3, 1)

        self.score_pool2_seg = nn.Conv2d(16, self.num_classes, 1)
        self.score_pool3_seg = nn.Conv2d(32, self.num_classes, 1)

        self.score_pool2_depth = nn.Conv2d(16, 1, 1)
        self.score_pool3_depth = nn.Conv2d(32, 1, 1)

        self.score_pool2_ae = nn.Conv2d(16, 3, 1)
        self.score_pool3_ae = nn.Conv2d(32, 3, 1)

        if self.only_encoder:
            if pretrained:
                print("Loading CNN weights from %s" % checkpoint_path)
                checkpoint = torch.load(
                    checkpoint_path, map_location={"cuda:0": "cpu"}
                )
                self.load_state_dict(checkpoint)

                if freeze_encoder:
                    for param in self.parameters():
                        param.requires_grad = False
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = (
                        m.kernel_size[0]
                        * m.kernel_size[1]
                        * (m.out_channels + m.in_channels)
                    )
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x):

        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)

        if self.only_encoder:
            return conv4.view(-1, 32 * 12 * 12)

        encoder_output = self.classifier(conv4)

        encoder_output_seg = self.encoder_seg(encoder_output)
        encoder_output_depth = self.encoder_depth(encoder_output)
        encoder_output_ae = self.encoder_ae(encoder_output)

        score_pool2_seg = self.score_pool2_seg(conv2)
        score_pool3_seg = self.score_pool3_seg(conv3)

        score_pool2_depth = self.score_pool2_depth(conv2)
        score_pool3_depth = self.score_pool3_depth(conv3)

        score_pool2_ae = self.score_pool2_ae(conv2)
        score_pool3_ae = self.score_pool3_ae(conv3)

        score_seg = F.interpolate(
            encoder_output_seg,
            score_pool3_seg.size()[2:],
            mode="bilinear",
            align_corners=True,
        )
        score_seg += score_pool3_seg
        score_seg = F.interpolate(
            score_seg,
            score_pool2_seg.size()[2:],
            mode="bilinear",
            align_corners=True,
        )
        score_seg += score_pool2_seg
        out_seg = F.interpolate(
            score_seg, x.size()[2:], mode="bilinear", align_corners=True
        )

        score_depth = F.interpolate(
            encoder_output_depth,
            score_pool3_depth.size()[2:],
            mode="bilinear",
            align_corners=True,
        )
        score_depth += score_pool3_depth
        score_depth = F.interpolate(
            score_depth,
            score_pool2_depth.size()[2:],
            mode="bilinear",
            align_corners=True,
        )
        score_depth += score_pool2_depth
        out_depth = torch.sigmoid(
            F.interpolate(
                score_depth, x.size()[2:], mode="bilinear", align_corners=True
            )
        )

        score_ae = F.interpolate(
            encoder_output_ae,
            score_pool3_ae.size()[2:],
            mode="bilinear",
            align_corners=True,
        )
        score_ae += score_pool3_ae
        score_ae = F.interpolate(
            score_ae,
            score_pool2_ae.size()[2:],
            mode="bilinear",
            align_corners=True,
        )
        score_ae += score_pool2_ae
        out_ae = torch.sigmoid(
            F.interpolate(
                score_ae, x.size()[2:], mode="bilinear", align_corners=True
            )
        )

        return out_seg, out_depth, out_ae
