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
- Download the Matterport 3D **scene dataset** and **task dataset** and place them in the appropriate folders (relevant information in repository's [README](/README.md)).

---

## EQA-CNN-Pretrain model

### Information:
This is an encoder-decoder network that takes RGB input and generates an RGB reconstruction, a depth map and a a Segmentation map. The encoder from this network is extracted and used as a feature extractor for subsequent VQA and NAV trainers.

(more information about network in Appendix B of [EQA paper](https://embodiedqa.org/paper.pdf)).

### Configuration:

Configuration for training the VQA (answering) model can be found in `habitat_baselines/config/eqa/il_eqa_cnn_pretrain.yaml`.

### Train:

```
python -u -m habitat_baselines.run \
  --config-name=eqa/il_eqa_cnn_pretrain.yaml
```

Training checkpoints are by default stored in `data/eqa/eqa_cnn_pretrain/checkpoints`.

### Eval:

```
python -u -m habitat_baselines.run \
  --config-name=eqa/il_eqa_cnn_pretrain.yaml \
  habitat_baselines.evaluate=True
```

Results from evaluation are stored in `data/eqa/eqa_cnn_pretrain/results/val`.


### Pre-trained model

Pre-trained EQA-CNN-Pretrain model can be downloaded from [here](https://drive.google.com/drive/folders/1yO8Pnyt-oxqAz0ozxwyI3OcaFRiKZKgd?usp=sharing).

After downloading the pre-trained model, it's path needs to be added to the config file's `eval_ckpt_path_dir` parameter for evaluation.

### Example results:

<img src="https://user-images.githubusercontent.com/24846546/76339759-6f788b00-62f2-11ea-90e0-a8ac16c34f76.jpg" width=40%>

---

## Visual Question Answering (VQA) model-

### Information:
The VQA model is responsible for predicting an answer based on the input question and a series of RGB images. The network first encodes images from the scene using the pre-trained EQA-CNN encoder mentioned above.

(more information about network can be found in the [paper](https://embodiedqa.org/paper.pdf)).

### Configuration:

Configuration for training the VQA (answering) model can be found in `habitat_baselines/config/eqa/il_vqa.yaml`.

The VQA trainer picks the EQA CNN pre-trained encoder checkpoint by default from `data/eqa/eqa_cnn_pretrain/checkpoints/epoch_5.ckpt`. If you want to use a different checkpoint for the EQA CNN encoder, the corresponding path can be changed in the aforementioned config file's `eqa_cnn_pretrain_ckpt_path` parameter.

### Train:

```
python -u -m habitat_baselines.run \
  --config-name=eqa/il_vqa.yaml
```

Training checkpoints are by default stored in `data/eqa/vqa/checkpoints`.

### Pre-trained model

Pre-trained VQA model can be downloaded from [here](https://drive.google.com/file/d/1frhIlgF1BpBT_vnRt7J5txlnfvgwXLeq/view?usp=sharing).

After downloading the pre-trained model, add its path to the config file's `eval_ckpt_path_dir` parameter for evaluation.

### Eval:

```
python -u -m habitat_baselines.run \
  --config-name=/eqa/il_vqa.yaml \
  habitat_baselines.evaluate=True
```

Results from evaluation are stored in `data/eqa/vqa/results/val`.

### Example results:

![](https://user-images.githubusercontent.com/24846546/75141155-464bde00-56e8-11ea-9f2e-ca346440e1d2.jpg)
![](https://user-images.githubusercontent.com/24846546/75141287-8e6b0080-56e8-11ea-8045-b4c4521954b2.jpg)

----

## NAV model (PACMAN)

### Information:
The NAV model (known as *PACMAN*) predicts the actions required to navigate the environment to the required destination based on question and RGB scene input.

(more information about network can be found in the [paper](https://embodiedqa.org/paper.pdf)).

### Configuration:

Configuration for training the NAV-PACMAN model can be found in `habitat_baselines/config/eqa/il_pacman_nav.yaml`.
The trainer also picks the EQA CNN pre-trained encoder checkpoint by default from `data/eqa/eqa_cnn_pretrain/checkpoints/epoch_5.ckpt`.

### Train:

```
python -u -m habitat_baselines.run \
  --config-name=eqa/il_pacman_nav.yaml
```

Training checkpoints are by default stored in `data/eqa/nav/checkpoints`.


### Eval:

```
python -u -m habitat_baselines.run \
  --config-name=eqa/il_pacman_nav.yaml \
  habitat_baselines.evaluate=True
```

Results from evaluation are stored in `data/eqa/nav/results/val`.

### Example results:

![](https://user-images.githubusercontent.com/24846546/78616220-2d942380-7863-11ea-9092-34a760352555.gif) ![](https://user-images.githubusercontent.com/24846546/78616221-2ec55080-7863-11ea-987b-2fdc2a802f24.gif) ![](https://user-images.githubusercontent.com/24846546/78616897-2cfc8c80-7865-11ea-8a4c-0afdfefea49c.gif)
