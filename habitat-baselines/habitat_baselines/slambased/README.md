### Handcrafted agent baseline adopted from the paper "Benchmarking Classic and Learned Navigation in Complex 3D Environments"

Project website: https://sites.google.com/view/classic-vs-learned-navigation
Paper: https://arxiv.org/abs/1901.10915

<p align="center">
  <img src="data/slam-based-agent.png">
</p>

If you use this code or the provided environments in your research, please cite the following:

    @ARTICLE{Navigation2019,
           author = {{Mishkin}, Dmytro and {Dosovitskiy}, Alexey and {Koltun}, Vladlen},
            title = "{Benchmarking Classic and Learned Navigation in Complex 3D Environments}",
             year = 2019,
            month = Jan,
    archivePrefix = {arXiv},
           eprint = {1901.10915},
    }



## Dependencies:

- conda
- numpy
- pytorch
- ORBSLAM2


## Tested with:
- Ubuntu 16.04
- python 3.7
- pytorch 0.4, 1.0


- Install Anaconda https://www.anaconda.com/download/#linux

- Install dependencies via ./install_deps.sh.  It should install everything except the datasets.

Simple example of working with agents is shown in (../handcrafted-agent-example.ipynb)
