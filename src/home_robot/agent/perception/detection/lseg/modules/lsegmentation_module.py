from argparse import ArgumentParser
import random
import time

from data import get_available_datasets, get_dataset
from encoding.nn import SegmentationLosses
from encoding.utils import (
    batch_intersection_union,
    batch_pix_accuracy,
    SegmentationMetric,
)
import pytorch_lightning as pl
import torch

# add mixed precision
import torch.cuda.amp as amp


class LSegmentationModule(pl.LightningModule):
    def __init__(self, data_path, dataset, batch_size, base_lr, max_epochs, **kwargs):
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.base_lr = base_lr / 16 * batch_size
        self.lr = self.base_lr

        self.epochs = max_epochs
        self.other_kwargs = kwargs
        self.enabled = False  # True mixed precision will make things complicated and leading to NAN error
        self.scaler = amp.GradScaler(enabled=self.enabled)

    def forward(self, x):
        return self.net(x)

    def evaluate(self, x, target=None):
        pred = self.net.forward(x)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(pred.data, target.data, self.nclass)

        return correct, labeled, inter, union

    def evaluate_random(self, x, labelset, target=None):
        pred = self.net.forward(x, labelset)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(pred.data, target.data, self.nclass)

        return correct, labeled, inter, union

    def training_step(self, batch, batch_nb):
        img, target = batch
        with amp.autocast(enabled=self.enabled):
            out = self(img)
            multi_loss = isinstance(out, tuple)
            if multi_loss:
                loss = self.criterion(*out, target)
            else:
                loss = self.criterion(out, target)
            loss = self.scaler.scale(loss)
        final_output = out[0] if multi_loss else out
        train_pred, train_gt = self._filter_invalid(final_output, target)
        if train_gt.nelement() != 0:
            self.train_accuracy(train_pred, train_gt)
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outs):
        self.log("train_acc_epoch", self.train_accuracy.compute())

    def validation_step(self, batch, batch_nb):
        img, target = batch
        out = self(img)
        multi_loss = isinstance(out, tuple)
        if multi_loss:
            val_loss = self.criterion(*out, target)
        else:
            val_loss = self.criterion(out, target)
        final_output = out[0] if multi_loss else out
        valid_pred, valid_gt = self._filter_invalid(final_output, target)
        self.val_iou.update(target, final_output)
        pixAcc, iou = self.val_iou.get()
        self.log("val_loss_step", val_loss)
        self.log("pix_acc_step", pixAcc)
        self.log(
            "val_acc_step",
            self.val_accuracy(valid_pred, valid_gt),
        )
        self.log("val_iou", iou)

    def validation_epoch_end(self, outs):
        pixAcc, iou = self.val_iou.get()
        self.log("val_acc_epoch", self.val_accuracy.compute())
        self.log("val_iou_epoch", iou)
        self.log("pix_acc_epoch", pixAcc)

        self.val_iou.reset()

    def _filter_invalid(self, pred, target):
        valid = target != self.other_kwargs["ignore_index"]
        _, mx = torch.max(pred, dim=1)
        return mx[valid], target[valid]

    def configure_optimizers(self):
        params_list = [
            {"params": self.net.pretrained.parameters(), "lr": self.base_lr},
        ]
        if hasattr(self.net, "scratch"):
            print("Found output scratch")
            params_list.append(
                {"params": self.net.scratch.parameters(), "lr": self.base_lr * 10}
            )
        if hasattr(self.net, "auxlayer"):
            print("Found auxlayer")
            params_list.append(
                {"params": self.net.auxlayer.parameters(), "lr": self.base_lr * 10}
            )
        if hasattr(self.net, "scale_inv_conv"):
            print(self.net.scale_inv_conv)
            print("Found scaleinv layers")
            params_list.append(
                {
                    "params": self.net.scale_inv_conv.parameters(),
                    "lr": self.base_lr * 10,
                }
            )
            params_list.append(
                {"params": self.net.scale2_conv.parameters(), "lr": self.base_lr * 10}
            )
            params_list.append(
                {"params": self.net.scale3_conv.parameters(), "lr": self.base_lr * 10}
            )
            params_list.append(
                {"params": self.net.scale4_conv.parameters(), "lr": self.base_lr * 10}
            )

        if self.other_kwargs["midasproto"]:
            print("Using midas optimization protocol")

            opt = torch.optim.Adam(
                params_list,
                lr=self.base_lr,
                betas=(0.9, 0.999),
                weight_decay=self.other_kwargs["weight_decay"],
            )
            sch = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda x: pow(1.0 - x / self.epochs, 0.9)
            )

        else:
            opt = torch.optim.SGD(
                params_list,
                lr=self.base_lr,
                momentum=0.9,
                weight_decay=self.other_kwargs["weight_decay"],
            )
            sch = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda x: pow(1.0 - x / self.epochs, 0.9)
            )
        return [opt], [sch]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=16,
            worker_init_fn=lambda x: random.seed(time.time() + x),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=16,
        )

    def get_trainset(self, dset, augment=False, **kwargs):
        print(kwargs)
        if augment == True:
            mode = "train_x"
        else:
            mode = "train"

        print(mode)
        dset = get_dataset(
            dset,
            root=self.data_path,
            split="train",
            mode=mode,
            transform=self.train_transform,
            **kwargs
        )

        self.num_classes = dset.num_class
        self.train_accuracy = pl.metrics.Accuracy()

        return dset

    def get_valset(self, dset, augment=False, **kwargs):
        self.val_accuracy = pl.metrics.Accuracy()
        self.val_iou = SegmentationMetric(self.num_classes)

        if augment == True:
            mode = "val_x"
        else:
            mode = "val"

        print(mode)
        return get_dataset(
            dset,
            root=self.data_path,
            split="val",
            mode=mode,
            transform=self.val_transform,
            **kwargs
        )

    def get_criterion(self, **kwargs):
        return SegmentationLosses(
            se_loss=kwargs["se_loss"],
            aux=kwargs["aux"],
            nclass=self.num_classes,
            se_weight=kwargs["se_weight"],
            aux_weight=kwargs["aux_weight"],
            ignore_index=kwargs["ignore_index"],
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--data_path", type=str, help="path where dataset is stored"
        )
        parser.add_argument(
            "--dataset",
            choices=get_available_datasets(),
            default="ade20k",
            help="dataset to train on",
        )
        parser.add_argument(
            "--batch_size", type=int, default=16, help="size of the batches"
        )
        parser.add_argument(
            "--base_lr", type=float, default=0.004, help="learning rate"
        )
        parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
        parser.add_argument(
            "--weight_decay", type=float, default=1e-4, help="weight_decay"
        )
        parser.add_argument(
            "--aux", action="store_true", default=False, help="Auxilary Loss"
        )
        parser.add_argument(
            "--aux-weight",
            type=float,
            default=0.2,
            help="Auxilary loss weight (default: 0.2)",
        )
        parser.add_argument(
            "--se-loss",
            action="store_true",
            default=False,
            help="Semantic Encoding Loss SE-loss",
        )
        parser.add_argument(
            "--se-weight", type=float, default=0.2, help="SE-loss weight (default: 0.2)"
        )

        parser.add_argument(
            "--midasproto", action="store_true", default=False, help="midasprotocol"
        )

        parser.add_argument(
            "--ignore_index",
            type=int,
            default=-1,
            help="numeric value of ignore label in gt",
        )
        parser.add_argument(
            "--augment",
            action="store_true",
            default=False,
            help="Use extended augmentations",
        )

        return parser
