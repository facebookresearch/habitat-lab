Imitation Learning (IL)
=======================

#### Embodied Question Answering

> based on EmbodiedQA (Das et al. CVPR 2018) implementation.

**Paper:** https://embodiedqa.org/paper.pdf
**Code:** https://github.com/facebookresearch/EmbodiedQA

The implementation consists of first independently training the -
- NAV model (for navigating to the required destination based on question and image input)
- VQA model (for predicting answer based on question and image input)

followed by fine-tuning the NAV model.

> "We employ a two-stage training process. First, the navigation and answering modules are independently trained using imitation/supervised learning on automatically generated expert demonstrations of navigation. Second, the navigation architecture is fine-tuned .."

### Pre-requisites:

- Habitat-sim and Habitat-api installation.
- Download the Matterport 3D **scene dataset** and **task dataset** and place them in the appropriate folders (relevant information in repository's [README](https://github.com/facebookresearch/habitat-api/blob/master/README.md)).
- **Pretrained CNN**: The authors of the paper have provided a pre-trained feature extractor for navigation and question answering. The CNN is available for download [here](https://www.dropbox.com/s/ju1zw4iipxlj966/03_13_h3d_hybrid_cnn.pt).
The training code expects the checkpoint to be in `habitat_baselines/il/models`.

### VQA model (answering module)- 

#### Configuration:

Configuration for training the VQA (answering) model can be found in `habitat_baselines/config/eqa/il_vqa.yaml`.

#### Train:

```
 python -u habitat_baselines/run.py --exp-config habitat_baselines/config/eqa/il_vqa.yaml --run-type train
```

Training checkpoints are by default stored in `data/vqa/checkpoints`.

#### Eval:

```
 python -u habitat_baselines/run.py --exp-config habitat_baselines/config/eqa/il_vqa.yaml --run-type eval
```

Results from evaluation are stored in `data/vqa/results/val`.

##### Example results:

![](https://user-images.githubusercontent.com/24846546/74877092-beb74580-535c-11ea-9199-c04985d1eed2.jpg)
![](https://user-images.githubusercontent.com/24846546/74877214-f4f4c500-535c-11ea-81f3-43924a072f4a.jpg)