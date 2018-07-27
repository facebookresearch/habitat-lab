The Embodied Agent Suite (TEAS)
==============================

Suite for training and benchmarking embodied agents.

**See the discussion in the [Quip folder](https://fb.quip.com/vAirA8l2Qrxv)**

## Intro

The Embodied Agent Suite aims to provide utilities to make it easier to train Embodied Agents in simulated environments. It will be open-sourced to support an "Embodied Question Answering challenge" workshop (CVPR19). While it will support the EQA challenge, this suite provides support for a range of tasks beyond just EQA.

It borrows heavily from the [OpenAI gym](https://github.com/openai/gym) interface which is well thought-out. It also provides additional tools on top of this interface which support training embodied agents. 

This suite contains tools at multiple levels:
- Environment wrappers to provide a unified interface to different simulators (ESP, Gibson, Minos, and others)
- Tools for distributing environments over multiple processes, gpus, and machines
- An interface between OpenAI gym and models at the layer of physical sensors
- Tools for rewarding embodied agents: reward shaping, API calls to get shortest path, 
- and, of course, much more
