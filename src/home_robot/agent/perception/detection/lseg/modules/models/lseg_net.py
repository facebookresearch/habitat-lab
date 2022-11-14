import clip
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.patches as mpatches

from .lseg_blocks import _make_encoder, FeatureFusionBlock_custom, forward_vit, Interpolate


class depthwise_clipseg_conv(nn.Module):
    def __init__(self):
        super(depthwise_clipseg_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def depthwise_clipseg(self, x, channels):
        x = torch.cat([self.depthwise(x[:, i].unsqueeze(1)) for i in range(channels)], dim=1)
        return x

    def forward(self, x):
        channels = x.shape[1]
        out = self.depthwise_clipseg(x, channels)
        return out


class depthwise_conv(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # support for 4D tensor with NCHW
        C, H, W = x.shape[1:]
        x = x.reshape(-1, 1, H, W)
        x = self.depthwise(x)
        x = x.view(-1, C, H, W)
        return x


class depthwise_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation="relu"):
        super(depthwise_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()

    def forward(self, x, act=True):
        x = self.depthwise(x)
        if act:
            x = self.activation(x)
        return x


class bottleneck_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation="relu"):
        super(bottleneck_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()

    def forward(self, x, act=True):
        sum_layer = x.max(dim=1, keepdim=True)[0]
        x = self.depthwise(x)
        x = x + sum_layer
        if act:
            x = self.activation(x)
        return x


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.
        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        activation=nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class LSeg(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="clip_vitl16_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs,
    ):
        super(LSeg, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "clip_vitl16_384": [5, 11, 17, 23],
            "clipRN50x16_vitl16_384": [5, 11, 17, 23],
            "clip_vitb32_384": [2, 5, 8, 11],
        }

        # Instantiate backbone and reassemble blocks
        self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        if backbone in ["clipRN50x16_vitl16_384"]:
            self.out_c = 768
        else:
            self.out_c = 512
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

        self.arch_option = kwargs["arch_option"]
        if self.arch_option == 1:
            self.scratch.head_block = bottleneck_block(activation=kwargs["activation"])
            self.block_depth = kwargs["block_depth"]
        elif self.arch_option == 2:
            self.scratch.head_block = depthwise_block(activation=kwargs["activation"])
            self.block_depth = kwargs["block_depth"]

        self.scratch.output_conv = head

        self.text = clip.tokenize(self.labels)

    def forward(self, x, labelset=""):
        if labelset == "":
            text = self.text
        else:
            text = clip.tokenize(labelset)

        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        text = text.to(x.device)
        self.logit_scale = self.logit_scale.to(x.device)
        text_features = self.clip_pretrained.encode_text(text)

        image_features = self.scratch.head1(path_1)

        imshape = image_features.shape
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, self.out_c)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        pixel_encoding = self.logit_scale * image_features.half()

        logits_per_image = pixel_encoding @ text_features.t()

        out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)

        if self.arch_option in [1, 2]:
            for _ in range(self.block_depth - 1):
                out = self.scratch.head_block(out)
            out = self.scratch.head_block(out, False)

        out = self.scratch.output_conv(out)

        return out


class LSegNet(LSeg):
    """Network for semantic segmentation."""

    def __init__(self, labels, path=None, scale_factor=0.5, crop_size=480, **kwargs):
        kwargs["use_bn"] = True

        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.labels = labels

        head = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)


class LSegEnc(BaseModel):
    """
    LSeg encoder network.
    """

    def __init__(
        self,
        head,
        features=256,
        backbone="clip_vitl16_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs,
    ):
        super(LSegEnc, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "clip_vitl16_384": [5, 11, 17, 23],
            "clipRN50x16_vitl16_384": [5, 11, 17, 23],
            "clip_vitb32_384": [2, 5, 8, 11],
        }

        # Instantiate backbone and reassemble blocks
        self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        if backbone in ["clipRN50x16_vitl16_384"]:
            self.out_c = 768
        else:
            self.out_c = 512
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

        self.arch_option = kwargs["arch_option"]
        if self.arch_option == 1:
            self.scratch.head_block = bottleneck_block(activation=kwargs["activation"])
            self.block_depth = kwargs["block_depth"]
        elif self.arch_option == 2:
            self.scratch.head_block = depthwise_block(activation=kwargs["activation"])
            self.block_depth = kwargs["block_depth"]

        self.scratch.output_conv = head

    def forward(self, x):
        """Encode RGB image to CLIP features."""
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        self.logit_scale = self.logit_scale.to(x.device)

        image_features = self.scratch.head1(path_1)

        imshape = image_features.shape
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, self.out_c)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        pixel_encoding = self.logit_scale * image_features.half()
        pixel_encoding = pixel_encoding.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)
        pixel_encoding = self.scratch.output_conv(pixel_encoding)

        return pixel_encoding


def get_new_palette(num_cls):
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab > 0:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3
    return palette


def get_new_mask_palette(npimg, new_palette, out_label_flag=False,
                         labels=None, ignore_ids_list=[]):
    """Get image color palette for visualizing masks."""
    out_img = Image.fromarray(npimg.squeeze().astype("uint8"))
    out_img.putpalette(new_palette)

    if out_label_flag:
        assert labels is not None
        u_index = np.unique(npimg)
        patches = []
        for i, index in enumerate(u_index):
            if index in ignore_ids_list:
                continue
            label = labels[index]
            cur_color = [
                new_palette[index * 3] / 255.0,
                new_palette[index * 3 + 1] / 255.0,
                new_palette[index * 3 + 2] / 255.0,
            ]
            red_patch = mpatches.Patch(color=cur_color, label=label)
            patches.append(red_patch)
    return out_img, patches


class LSegEncDecNet(LSegEnc):
    """LSeg encoder & decoder wrapper."""

    def __init__(self,
                 path=None,
                 scale_factor=0.5,
                 norm_mean=[0.5, 0.5, 0.5],
                 norm_std=[0.5, 0.5, 0.5],
                 visualize=True,
                 **kwargs):
        kwargs["use_bn"] = True

        self.scale_factor = scale_factor
        self.visualize = visualize
        self.transform = transforms.Compose(
            [
                transforms.Normalize(norm_mean, norm_std),
            ]
        )

        head = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)

    def encode(self, images) -> torch.Tensor:
        """Encode RGB images to CLIP pixel features.

        Arguments:
            images: images of shape (batch_size, H, W, 3) (in RGB order)

        Returns:
            pixel_features: CLIP pixel features of shape (batch_size, 512, H, W)
        """
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        device = next(self.parameters()).device
        images = images.to(device).permute((0, 3, 1, 2))
        images = self.transform(images / 255.0)
        return self.forward(images)

    def decode(self,
               pixel_features: torch.Tensor,
               labels: Optional[List[str]] = None
               ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        """Decode CLIP pixel features to text labels.

        Arguments:
            pixel_features: CLIP pixel features of shape (batch_size, 512, H, W)
            labels: set of text labels

        Returns:
            one_hot_predictions: one hot segmentation predictions of shape
             (batch_size, H, W, len(labels))
            visualizations: prediction visualization images of shape
             (batch_size, H, W, 3) if self.visualize=True else None
        """
        device = next(self.parameters()).device
        text = clip.tokenize(labels).to(device)
        text_features = self.clip_pretrained.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.float()

        label_scores = pixel_features.permute((0, 2, 3, 1)) @ text_features.T

        predictions = torch.argmax(label_scores, dim=-1)
        one_hot_predictions = torch.eye(len(labels)).to(device)[predictions]

        if self.visualize:
            visualizations = []
            for prediction in predictions.cpu().numpy():
                new_palette = get_new_palette(len(labels))
                mask, patches = get_new_mask_palette(
                    prediction, new_palette, out_label_flag=True, labels=labels)
                mask = mask.convert("RGB")
                visualizations.append(np.array(mask))
            visualizations = np.stack(visualizations)
        else:
            visualizations = None

        return one_hot_predictions, visualizations
