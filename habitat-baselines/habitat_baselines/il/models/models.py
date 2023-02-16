# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Iterable, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from habitat import logger

HiddenState = Union[Tensor, Tuple[Tensor, Tensor], None]


def build_mlp(
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int,
    use_batchnorm: bool = False,
    dropout: float = 0,
    add_sigmoid: bool = True,
):
    layers: List[nn.Module] = []
    D = input_dim
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(input_dim))
    for dim in hidden_dims:
        layers.append(nn.Linear(D, dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(dim))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.ReLU(inplace=True))
        D = dim
    layers.append(nn.Linear(D, output_dim))

    if add_sigmoid:
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class MultitaskCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 41,
        only_encoder: bool = False,
        pretrained: bool = True,
        checkpoint_path: str = "data/eqa/eqa_cnn_pretrain/checkpoints/epoch_5.ckpt",
        freeze_encoder: bool = False,
    ) -> None:
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
                logger.info(
                    "Loading CNN weights from {}".format(checkpoint_path)
                )
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

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
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


class QuestionLstmEncoder(nn.Module):
    def __init__(
        self,
        token_to_idx: Dict,
        wordvec_dim: int = 64,
        rnn_dim: int = 64,
        rnn_num_layers: int = 2,
        rnn_dropout: float = 0,
    ) -> None:
        super(QuestionLstmEncoder, self).__init__()

        self.token_to_idx = token_to_idx
        self.NULL = token_to_idx["<pad>"]
        self.START = token_to_idx["<s>"]
        self.END = token_to_idx["</s>"]

        self.embed = nn.Embedding(len(token_to_idx), wordvec_dim)
        self.rnn = nn.LSTM(
            wordvec_dim,
            rnn_dim,
            rnn_num_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: Tensor) -> Tensor:
        N, T = x.size()
        idx = torch.LongTensor(N).fill_(T - 1)

        # Find the last non-null element in each sequence
        idx = (x != self.NULL).long().sum(-1) - 1
        idx = idx.type_as(x.data).long()
        idx.requires_grad = False

        hs, _ = self.rnn(self.embed(x.long()))

        idx = idx.view(N, 1, 1).expand(N, 1, hs.size(2))
        H = hs.size(2)
        return hs.gather(1, idx).view(N, H)


class VqaLstmCnnAttentionModel(nn.Module):
    def __init__(
        self,
        q_vocab: Dict,
        ans_vocab: Dict,
        eqa_cnn_pretrain_ckpt_path: str,
        freeze_encoder: bool = False,
        image_feat_dim: int = 64,
        question_wordvec_dim: int = 64,
        question_hidden_dim: int = 64,
        question_num_layers: int = 2,
        question_dropout: float = 0.5,
        fc_use_batchnorm: bool = False,
        fc_dropout: float = 0.5,
        fc_dims: Iterable[int] = (64,),
    ) -> None:
        super(VqaLstmCnnAttentionModel, self).__init__()

        cnn_kwargs = {
            "num_classes": 41,
            "only_encoder": True,
            "pretrained": True,
            "checkpoint_path": eqa_cnn_pretrain_ckpt_path,
            "freeze_encoder": freeze_encoder,
        }
        self.cnn = MultitaskCNN(**cnn_kwargs)  # type:ignore
        self.cnn_fc_layer = nn.Sequential(
            nn.Linear(32 * 12 * 12, 64), nn.ReLU(), nn.Dropout(p=0.5)
        )

        q_rnn_kwargs = {
            "token_to_idx": q_vocab,
            "wordvec_dim": question_wordvec_dim,
            "rnn_dim": question_hidden_dim,
            "rnn_num_layers": question_num_layers,
            "rnn_dropout": question_dropout,
        }
        self.q_rnn = QuestionLstmEncoder(**q_rnn_kwargs)  # type:ignore

        self.img_tr = nn.Sequential(nn.Linear(64, 64), nn.Dropout(p=0.5))

        self.ques_tr = nn.Sequential(nn.Linear(64, 64), nn.Dropout(p=0.5))

        classifier_kwargs = {  # type:ignore
            "input_dim": 64,
            "hidden_dims": fc_dims,
            "output_dim": len(ans_vocab),
            "use_batchnorm": True,
            "dropout": fc_dropout,
            "add_sigmoid": False,
        }
        self.classifier = build_mlp(**classifier_kwargs)  # type:ignore

        self.att = nn.Sequential(
            nn.Tanh(), nn.Dropout(p=0.5), nn.Linear(128, 1)
        )

    def forward(
        self, images: Tensor, questions: Tensor
    ) -> Tuple[Tensor, Tensor]:
        N, T, _, _, _ = images.size()
        # bs x 5 x 3 x 256 x 256
        img_feats = self.cnn(
            images.contiguous().view(
                -1, images.size(2), images.size(3), images.size(4)
            )
        )

        img_feats = self.cnn_fc_layer(img_feats)

        img_feats_tr = self.img_tr(img_feats)
        ques_feats = self.q_rnn(questions)

        ques_feats_repl = ques_feats.view(N, 1, -1).repeat(1, T, 1)
        ques_feats_repl = ques_feats_repl.view(N * T, -1)

        ques_feats_tr = self.ques_tr(ques_feats_repl)

        ques_img_feats = torch.cat([ques_feats_tr, img_feats_tr], 1)

        att_feats = self.att(ques_img_feats)
        att_probs = F.softmax(att_feats.view(N, T), dim=1)
        att_probs2 = att_probs.view(N, T, 1).repeat(1, 1, 64)

        att_img_feats = torch.mul(att_probs2, img_feats.view(N, T, 64))
        att_img_feats = torch.sum(att_img_feats, dim=1)

        mul_feats = torch.mul(ques_feats, att_img_feats)

        scores = self.classifier(mul_feats)

        return scores, att_probs


class MaskedNLLCriterion(nn.Module):
    def __init__(self) -> None:
        super(MaskedNLLCriterion, self).__init__()

    def forward(self, inp: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        logprob_select = torch.gather(inp, 1, target.long())
        out = torch.masked_select(logprob_select, mask)
        loss = -torch.sum(out) / mask.float().sum()
        return loss


class NavPlannerControllerModel(nn.Module):
    def __init__(
        self,
        q_vocab: Dict,
        num_output: int = 4,
        question_wordvec_dim: int = 64,
        question_hidden_dim: int = 64,
        question_num_layers: int = 2,
        question_dropout: float = 0.5,
        planner_rnn_image_feat_dim: int = 128,
        planner_rnn_action_embed_dim: int = 32,
        planner_rnn_type: str = "GRU",
        planner_rnn_hidden_dim: int = 1024,
        planner_rnn_num_layers: int = 1,
        planner_rnn_dropout: float = 0,
        controller_fc_dims: Iterable[int] = (256,),
    ) -> None:
        super(NavPlannerControllerModel, self).__init__()

        self.cnn_fc_layer = nn.Sequential(
            nn.Linear(32 * 12 * 12, planner_rnn_image_feat_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        q_rnn_kwargs = {
            "token_to_idx": q_vocab,
            "wordvec_dim": question_wordvec_dim,
            "rnn_dim": question_hidden_dim,
            "rnn_num_layers": question_num_layers,
            "rnn_dropout": question_dropout,
        }
        self.q_rnn = QuestionLstmEncoder(**q_rnn_kwargs)  # type:ignore
        self.ques_tr = nn.Sequential(
            nn.Linear(question_hidden_dim, question_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.planner_nav_rnn = NavRnn(
            image_input=True,
            image_feat_dim=planner_rnn_image_feat_dim,
            question_input=True,
            question_embed_dim=question_hidden_dim,
            action_input=True,
            action_embed_dim=planner_rnn_action_embed_dim,
            num_actions=num_output,
            rnn_type=planner_rnn_type,
            rnn_hidden_dim=planner_rnn_hidden_dim,
            rnn_num_layers=planner_rnn_num_layers,
            rnn_dropout=planner_rnn_dropout,
            return_states=True,
        )

        controller_kwargs = {
            "input_dim": planner_rnn_image_feat_dim
            + planner_rnn_action_embed_dim
            + planner_rnn_hidden_dim,
            "hidden_dims": controller_fc_dims,
            "output_dim": 2,
            "add_sigmoid": 0,
        }
        self.controller = build_mlp(**controller_kwargs)  # type:ignore

    def forward(
        self,
        questions: Tensor,
        planner_img_feats: Tensor,
        planner_actions_in: Tensor,
        planner_action_lengths: Tensor,
        planner_hidden_index: Tensor,
        controller_img_feats: Tensor,
        controller_actions_in: Tensor,
        controller_action_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        N_p, T_p, _ = planner_img_feats.size()

        planner_img_feats = self.cnn_fc_layer(planner_img_feats)
        controller_img_feats = self.cnn_fc_layer(controller_img_feats)

        ques_feats = self.q_rnn(questions)
        ques_feats = self.ques_tr(ques_feats)

        planner_states, planner_scores, planner_hidden = self.planner_nav_rnn(
            planner_img_feats,
            ques_feats,
            planner_actions_in,
            planner_action_lengths,
        )

        planner_hidden_index = planner_hidden_index[
            :, : controller_action_lengths.max()  # type:ignore
        ]
        controller_img_feats = controller_img_feats[
            :, : controller_action_lengths.max()  # type:ignore
        ]
        controller_actions_in = controller_actions_in[
            :, : controller_action_lengths.max()  # type:ignore
        ]

        N_c, T_c, _ = controller_img_feats.size()

        assert planner_hidden_index.max().item() <= planner_states.size(1)

        planner_hidden_index = (
            planner_hidden_index.contiguous()
            .view(N_p, planner_hidden_index.size(1), 1)
            .repeat(1, 1, planner_states.size(2))
        )

        controller_hidden_in = planner_states.gather(
            1, planner_hidden_index.long()
        )

        controller_hidden_in = controller_hidden_in.view(
            N_c * T_c, controller_hidden_in.size(2)
        )

        controller_img_feats = controller_img_feats.contiguous().view(
            N_c * T_c, -1
        )

        controller_actions_embed = self.planner_nav_rnn.action_embed(
            controller_actions_in.long()
        ).view(N_c * T_c, -1)

        controller_in = torch.cat(
            [
                controller_img_feats,
                controller_actions_embed,
                controller_hidden_in,
            ],
            1,
        )
        controller_scores = self.controller(controller_in)
        return planner_scores, controller_scores, planner_hidden

    def planner_step(
        self,
        questions: Tensor,
        img_feats: Tensor,
        actions_in: Tensor,
        planner_hidden: HiddenState,
    ) -> Tuple[Tensor, Tensor]:
        img_feats = self.cnn_fc_layer(img_feats)
        ques_feats = self.q_rnn(questions)
        ques_feats = self.ques_tr(ques_feats)
        planner_scores, planner_hidden = self.planner_nav_rnn.step_forward(
            img_feats, ques_feats, actions_in, planner_hidden
        )

        return planner_scores, planner_hidden

    def controller_step(
        self, img_feats: Tensor, actions_in: Tensor, hidden_in: Tensor
    ) -> Tensor:
        img_feats = self.cnn_fc_layer(img_feats)
        actions_embed = self.planner_nav_rnn.action_embed(actions_in)

        img_feats = img_feats.view(1, -1)
        actions_embed = actions_embed.view(1, -1)
        hidden_in = hidden_in.view(1, -1)

        controller_in = torch.cat([img_feats, actions_embed, hidden_in], 1)
        controller_scores = self.controller(controller_in)

        return controller_scores


class NavRnn(nn.Module):
    def __init__(
        self,
        image_input: bool = False,
        image_feat_dim: int = 128,
        question_input: bool = False,
        question_embed_dim: int = 128,
        action_input: bool = False,
        action_embed_dim: int = 32,
        num_actions: int = 4,
        mode: str = "sl",
        rnn_type: str = "LSTM",
        rnn_hidden_dim: int = 128,
        rnn_num_layers: int = 2,
        rnn_dropout: float = 0,
        return_states: bool = False,
    ) -> None:
        super(NavRnn, self).__init__()

        self.image_input = image_input
        self.image_feat_dim = image_feat_dim

        self.question_input = question_input
        self.question_embed_dim = question_embed_dim

        self.action_input = action_input
        self.action_embed_dim = action_embed_dim

        self.num_actions = num_actions

        self.rnn_type = rnn_type
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers

        self.return_states = return_states

        rnn_input_dim = 0
        if self.image_input is True:
            rnn_input_dim += image_feat_dim
            logger.info(
                "Adding input to {}: image, rnn dim: {}".format(
                    self.rnn_type, rnn_input_dim
                )
            )

        if self.question_input is True:
            rnn_input_dim += question_embed_dim
            logger.info(
                "Adding input to {}: question, rnn dim: {}".format(
                    self.rnn_type, rnn_input_dim
                )
            )

        if self.action_input is True:
            self.action_embed = nn.Embedding(num_actions, action_embed_dim)
            rnn_input_dim += action_embed_dim
            logger.info(
                "Adding input to {}: action, rnn dim: {}".format(
                    self.rnn_type, rnn_input_dim
                )
            )

        self.rnn = getattr(nn, self.rnn_type)(
            rnn_input_dim,
            self.rnn_hidden_dim,
            self.rnn_num_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )
        logger.info(
            "Building {} with hidden dim: {}".format(
                self.rnn_type, rnn_hidden_dim
            )
        )

        self.decoder = nn.Linear(self.rnn_hidden_dim, self.num_actions)

    def init_hidden(
        self, bsz: int
    ) -> Union[Tuple[Tensor, Tensor], Tensor, None]:
        weight = next(self.parameters()).data
        if self.rnn_type == "LSTM":
            return (
                weight.new(
                    self.rnn_num_layers, bsz, self.rnn_hidden_dim
                ).zero_(),
                weight.new(
                    self.rnn_num_layers, bsz, self.rnn_hidden_dim
                ).zero_(),
            )
        elif self.rnn_type == "GRU":
            return weight.new(
                self.rnn_num_layers, bsz, self.rnn_hidden_dim
            ).zero_()
        else:
            return None

    def forward(
        self,
        img_feats: Tensor,
        question_feats: Tensor,
        actions_in: Tensor,
        action_lengths: Tensor,
        hidden: bool = False,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        T: Union[int, bool] = False
        if self.image_input is True:
            N, T, _ = img_feats.size()
            input_feats = img_feats

        if self.question_input is True:
            N, D = question_feats.size()
            question_feats = question_feats.view(N, 1, D)
            if T is False:
                T = actions_in.size(1)
            question_feats = question_feats.repeat(1, T, 1)
            if len(input_feats) == 0:
                input_feats = question_feats
            else:
                input_feats = torch.cat([input_feats, question_feats], 2)

        if self.action_input is True:
            if len(input_feats) == 0:
                input_feats = self.action_embed(actions_in)
            else:
                input_feats = torch.cat(
                    [input_feats, self.action_embed(actions_in.long())], 2
                )

        packed_input_feats = pack_padded_sequence(
            input_feats, action_lengths, batch_first=True
        )
        packed_output, hidden_state = self.rnn(packed_input_feats)
        rnn_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.decoder(
            rnn_output.contiguous().view(
                rnn_output.size(0) * rnn_output.size(1), rnn_output.size(2)
            )
        )

        if self.return_states:
            return rnn_output, output, hidden_state
        else:
            return output, hidden_state

    def step_forward(
        self,
        img_feats: Tensor,
        question_feats: Tensor,
        actions_in: Tensor,
        hidden: HiddenState,
    ) -> Tuple[Tensor, Tensor]:
        T: Union[bool, int] = False
        if self.image_input is True:
            N, T, _ = img_feats.size()
            input_feats = img_feats

        if self.question_input is True:
            N, D = question_feats.size()
            question_feats = question_feats.view(N, 1, D)
            if T is False:
                T = actions_in.size(1)
            question_feats = question_feats.repeat(1, T, 1)
            if len(input_feats) == 0:
                input_feats = question_feats
            else:
                input_feats = torch.cat([input_feats, question_feats], 2)

        if self.action_input is True:
            if len(input_feats) == 0:
                input_feats = self.action_embed(actions_in)
            else:
                actions_in = actions_in.long()
                input_feats = torch.cat(
                    [input_feats, self.action_embed(actions_in)], 2
                )

        output, hidden = self.rnn(input_feats, hidden)
        output = self.decoder(
            output.contiguous().view(
                output.size(0) * output.size(1), output.size(2)
            )
        )

        return output, hidden
