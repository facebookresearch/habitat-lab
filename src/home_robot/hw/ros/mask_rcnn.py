import rospy
import torch
import torchvision
import numpy as np

from home_robot.hw.ros.camera import RosCamera
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
)  # , MaskRCNN_ResNet50_FPN_Weights

import matplotlib.pyplot as plt

import torchvision.transforms.functional as F


plt.rcParams["savefig.bbox"] = "tight"

COCO_CATEGORIES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
COLORS = np.random.random((len(COCO_CATEGORIES), 3))


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


class MaskRCNNServer(object):
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        # weights = MaskRCNN_ResNet50_FPN_Weights.DEFAUILT
        self.model = maskrcnn_resnet50_fpn(
            pretrained=True, progress=True, num_classes=len(COCO_CATEGORIES)
        ).to(self.device)
        self.model.eval()
        self.rgb_cam = RosCamera("/camera/color")
        self.rgb_cam.wait_for_image()

    def spin(self, rate=60):
        rate = rospy.Rate(rate)
        while not rospy.is_shutdown():
            rgb = self.rgb_cam.get()
            rgb = np.rot90(np.flipud(np.fliplr(rgb))).copy()
            rgb = torch.FloatTensor(rgb).to(self.device).permute(2, 0, 1)
            if rgb is None:
                rate.sleep()
                continue
            res = self.model([rgb])[0]
            print(res.keys())
            print(
                [
                    (COCO_CATEGORIES[int(l)], s)
                    for (s, l) in zip(res["scores"], res["labels"])
                ]
            )
            dbg = draw_bounding_boxes(rgb.to(torch.uint8).cpu(), res["boxes"], width=4)
            show(dbg)
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("mask_rcnn_node")
    server = MaskRCNNServer()
    server.spin()
