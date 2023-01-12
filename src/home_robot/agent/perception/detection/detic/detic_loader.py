import argparse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'third_party/CenterNet2/'))

from detectron2.config import get_cfg

from centernet.config import add_centernet_config
from .detic.config import add_detic_config
from .detic.predictor import VisualizationDemo


def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def get_detic(config_file=None, vocabulary="coco", custom_vocabulary="",
              checkpoint_file=None, sem_gpu_id=0, visualize=True):
    """Load trained Detic model for inference.

    Arguments:
        config_file: path to model config
        vocabulary: currently one of "coco" for indoor coco categories or "custom"
         for a custom set of categories
        custom_vocabulary: if vocabulary="custom", this should be a comma-separated
         list of classes (as a single string)
        checkpoint_file: path to model checkpoint
        sem_gpu_id: GPU ID to load the model on, -1 for CPU
        visualize: if True, the model will return inference prediction visualizations
    """
    assert vocabulary in ["coco", "custom"]
    if config_file is None:
        config_file = str(
            Path(__file__).resolve().parent /
            "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
        )
    if checkpoint_file is None:
        checkpoint_file = str(
            Path(__file__).resolve().parent /
            "models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
        )
    print(f"Loading Detic with config={config_file} and checkpoint={checkpoint_file}")

    string_args = f"""
        --config-file {config_file} --vocabulary {vocabulary}
        """

    if vocabulary == "custom":
        assert custom_vocabulary != ""
        string_args += f""" --custom_vocabulary {custom_vocabulary}"""

    string_args += f""" --opts MODEL.WEIGHTS {checkpoint_file}"""

    if sem_gpu_id == -1:
        string_args += """ MODEL.DEVICE cpu"""
    else:
        string_args += f""" MODEL.DEVICE cuda:{sem_gpu_id}"""

    string_args = string_args.split()
    args = get_parser().parse_args(string_args)
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg, args, visualize=visualize)
    return demo
