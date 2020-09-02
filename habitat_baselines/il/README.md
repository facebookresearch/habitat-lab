Imitation Learning (IL)
=======================

## Embodied Question Answering

**based on EmbodiedQA (Das et al. CVPR 2018) implementation.**

**Paper:** https://embodiedqa.org/paper.pdf

**Code:** https://github.com/facebookresearch/EmbodiedQA

The implementation consists of first independently training the -
- **[EQA-CNN-Pretrain](#eqa-cnn-pretrain-model)** for feature extraction
- **VQA model** (for predicting answer based on question and image input)
- **PACMAN (NAV) model** (for navigating to the required destination based on question and image input)

followed by fine-tuning the NAV model.

> "We employ a two-stage training process. First, the navigation and answering modules are independently trained using imitation/supervised learning on automatically generated expert demonstrations of navigation. Second, the navigation architecture is fine-tuned .."

## Pre-requisites:

- Habitat-sim and Habitat-api installation.
- Download the Matterport 3D **scene dataset** and **task dataset** and place them in the appropriate folders (relevant information in repository's [README](https://github.com/facebookresearch/habitat-api/blob/master/README.md)).

---

## EQA-CNN-Pretrain model

### Information:
This is an encoder-decoder network that takes RGB input and generates an RGB reconstruction, a depth map and a a Segmentation map. The encoder from this network is extracted and used as a feature extractor for subsequent VQA and NAV trainers.

(more information about network in Appendix B of [EQA paper](https://embodiedqa.org/paper.pdf)).

### Configuration:

Configuration for training the VQA (answering) model can be found in `habitat_baselines/config/eqa/il_eqa_cnn_pretrain.yaml`.

### Train:

```
 python -u habitat_baselines/run.py --exp-config habitat_baselines/config/eqa/il_eqa_cnn_pretrain.yaml --run-type train
```

Training checkpoints are by default stored in `data/eqa/eqa_cnn_pretrain/checkpoints`.

### Eval:

```
 python -u habitat_baselines/run.py --exp-config habitat_baselines/config/eqa/il_eqa_cnn_pretrain.yaml --run-type eval
```

Results from evaluation are stored in `data/eqa/eqa_cnn_pretrain/results/val`.


### Pre-trained model

Pre-trained EQA-CNN-Pretrain model can be downloaded from [here](https://drive.google.com/drive/folders/1yO8Pnyt-oxqAz0ozxwyI3OcaFRiKZKgd?usp=sharing).

After downloading the pre-trained model, it's path needs to be added to the config file's `EVAL_CKPT_PATH_DIR` parameter for evaluation.

### Example results:

<img src="https://user-images.githubusercontent.com/24846546/76339759-6f788b00-62f2-11ea-90e0-a8ac16c34f76.jpg" width=40%>

---

[ Code and information about other trainers to be added soon. ]
