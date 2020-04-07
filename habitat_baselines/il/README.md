Imitation Learning (IL)
=======================

## Embodied Question Answering

**based on EmbodiedQA (Das et al. CVPR 2018) implementation.**

**Paper:** https://embodiedqa.org/paper.pdf

**Code:** https://github.com/facebookresearch/EmbodiedQA

The implementation consists of first independently training the -
- **EDFE - Encoder Decoder for Feature Extraction** (the encoder from this later used as a feature extractor)
- **NAV model** (for navigating to the required destination based on question and image input)
- **VQA model** (for predicting answer based on question and image input)

followed by fine-tuning the NAV model.

> "We employ a two-stage training process. First, the navigation and answering modules are independently trained using imitation/supervised learning on automatically generated expert demonstrations of navigation. Second, the navigation architecture is fine-tuned .."

## Pre-requisites:

- Habitat-sim and Habitat-api installation.
- Download the Matterport 3D **scene dataset** and **task dataset** and place them in the appropriate folders (relevant information in repository's [README](https://github.com/facebookresearch/habitat-api/blob/master/README.md)).

---

## EDFE model (Encoder-Decoder for Feature Extraction)- 

### Information:
This is a encoder-decoder network that takes RGB input and generates an RGB reconstruction, a Depth map and a a Segmentation map. The encoder from this network is extracted and used as a frozen feature extractor for subsequent VQA and NAV trainers.

(more information about network in Appendix B of [paper](https://embodiedqa.org/paper.pdf)).

### Configuration:

Configuration for training the VQA (answering) model can be found in `habitat_baselines/config/eqa/il_edfe.yaml`.

### Train:

```
 python -u habitat_baselines/run.py --exp-config habitat_baselines/config/eqa/il_edfe.yaml --run-type train
```

Training checkpoints are by default stored in `data/eqa/edfe/checkpoints`.

### Eval:

```
 python -u habitat_baselines/run.py --exp-config habitat_baselines/config/eqa/il_edfe.yaml --run-type eval
```

Results from evaluation are stored in `data/eqa/edfe/results/val`.

### Example results:

Trained for 5 epochs on 100k images from MP3DEQA dataset episodes. ([Checkpoint link](https://drive.google.com/file/d/1onjsv8Y8PrAyUE8wp9oe-b8xpHKKtRJ1/view?usp=sharing))

<img src="https://user-images.githubusercontent.com/24846546/76339759-6f788b00-62f2-11ea-90e0-a8ac16c34f76.jpg" width=40%>

---

## VQA model (answering module)- 

### Information:
The VQA model is responsible for predicting an answer based on question and RGB image input. The network first encodes images from the scene using the pre-trained EDFE encoder mentioned above.

(more information about network can be found in the [paper](https://embodiedqa.org/paper.pdf)).

### Configuration:

Configuration for training the VQA (answering) model can be found in `habitat_baselines/config/eqa/il_vqa.yaml`.

The VQA trainer picks the EDFE encoder checkpoint by default from `data/eqa/edfe/checkpoints/epoch_5.ckpt`. If you haven't trained the EDFE model and want to use a different checkpoint, the corresponding path can be changed in the aforementioned configuration file's `EDFE_CKPT_PATH` parameter.

### Train:

```
 python -u habitat_baselines/run.py --exp-config habitat_baselines/config/eqa/il_vqa.yaml --run-type train
```

Training checkpoints are by default stored in `data/eqa/vqa/checkpoints`.

### Eval:

```
 python -u habitat_baselines/run.py --exp-config habitat_baselines/config/eqa/il_vqa.yaml --run-type eval
```

Results from evaluation are stored in `data/eqa/vqa/results/val`.

### Example results:

![](https://user-images.githubusercontent.com/24846546/75141155-464bde00-56e8-11ea-9f2e-ca346440e1d2.jpg)
![](https://user-images.githubusercontent.com/24846546/75141287-8e6b0080-56e8-11ea-8045-b4c4521954b2.jpg)

----

## NAV model

### Information:
The NAV model (known as *PACMAN*) predicts the actions required to navigate the environment to the required destination based on the question and RGB image input.

Here also, the pre-trained EDFE encoder is used for extracting features from scene images.

(more information about network can be found in the [paper](https://embodiedqa.org/paper.pdf)).

### Configuration:

Configuration for training the NAV model can be found in `habitat_baselines/config/eqa/il_nav.yaml`.

The NAV trainer picks the EDFE encoder checkpoint by default from `data/eqa/edfe/checkpoints/epoch_5.ckpt`. If you haven't trained the EDFE model and want to use a different checkpoint, the corresponding path can be changed in the aforementioned configuration file's `EDFE_CKPT_PATH` parameter.

### Train:

```
 python -u habitat_baselines/run.py --exp-config habitat_baselines/config/eqa/il_nav.yaml --run-type train
```

Training checkpoints are by default stored in `data/eqa/nav/checkpoints`. 


### Eval:

```
 python -u habitat_baselines/run.py --exp-config habitat_baselines/config/eqa/il_nav.yaml --run-type eval
```

Results from evaluation are stored in `data/eqa/nav/results/val`.

### Example results:

![](https://user-images.githubusercontent.com/24846546/78616220-2d942380-7863-11ea-9092-34a760352555.gif) ![](https://user-images.githubusercontent.com/24846546/78616221-2ec55080-7863-11ea-987b-2fdc2a802f24.gif) ![](https://user-images.githubusercontent.com/24846546/78616897-2cfc8c80-7865-11ea-8a4c-0afdfefea49c.gif)
