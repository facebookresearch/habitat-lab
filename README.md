# Adaptive Coordination in Social Embodied Rearrangement
This is the code for "Adaptive Coordination in Social Embodied Rearrangement".

<p align="center">
    <img width="85%" src="https://github.com/facebookresearch/habitat-lab/raw/social-eai/teaser_fig.png">
    <br />
    <a href="https://arxiv.org/abs/2306.00087">[Paper]</a>
</p>

**Note that this README only describes the details specific to "Adaptive Coordination in Social Embodied Rearrangement". For general details on Habitat, please see [the Habitat website](https://aihabitat.org) which includes tutorials, documentation, and examples.**

Check out [Habitat 3.0](https://aihabitat.org/habitat3/). Habitat 3.0 includes extends the functionality and tasks in this paper to include humanoid simulation, human-in-the-loop evaluation, more collaborative tasks, and more! Habitat 3.0 is actively maintained in the `main` branch of this repo.

## Installation
Install instructions for Linux using `conda`.
- `conda create -n hab -y python=3.9`
- `conda activate hab`
- `conda install -y habitat-sim=0.2.4 withbullet  headless -c conda-forge -c aihabitat`
- `cd` into this directory:
    - `pip install -e habitat-lab`
    - `pip install -e habitat-baselines`

# Running
All below commands are currently for Tidy House task. Run from the `habitat-baselines` directory.

- BDP: `python habitat_baselines/run.py --config-name=social_eai/bdp.yaml`
- PBT: `python habitat_baselines/run.py --config-name=social_eai/pbt.yaml`

## Citation
```
@inproceedings{szot2023adaptive,
 title     = {Adaptive Coordination for Social Embodied Rearrangement},
 author    = {Andrew Szot and Unnat Jain and Dhruv Batra and Zsolt Kira and Ruta Desai and Akshara Rai},
 year      = {2023},
 booktitle = {International Conference on Machine Learning}
}
```

## License
Habitat-Lab is MIT licensed. See the [LICENSE file](/LICENSE) for details.

The trained models and the task datasets are considered data derived from the correspondent scene datasets.

- Matterport3D based task datasets and trained models are distributed with [Matterport3D Terms of Use](http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf) and under [CC BY-NC-SA 3.0 US license](https://creativecommons.org/licenses/by-nc-sa/3.0/us/).
- Gibson based task datasets, the code for generating such datasets, and trained models are distributed with [Gibson Terms of Use](https://storage.googleapis.com/gibson_material/Agreement%20GDS%2006-04-18.pdf) and under [CC BY-NC-SA 3.0 US license](https://creativecommons.org/licenses/by-nc-sa/3.0/us/).
