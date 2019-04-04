# Base image
FROM nvidia/cudagl:9.0-runtime-ubuntu16.04

# Setup basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libegl1-mesa-dev \
    pkg-config \
    wget \
    zip \
    unzip &&\
    rm -rf /var/lib/apt/lists/*

# Install conda
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    chmod +x ~/miniconda.sh &&\
    ~/miniconda.sh -b -p /opt/conda &&\
    rm ~/miniconda.sh &&\
    /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl mkl-include &&\
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# Install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.13.4/cmake-3.13.4-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.13.4-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

# Conda environment
RUN conda create -n habitat python=3.6

# Setup habtiat-sim
RUN git clone https://github.com/facebookresearch/habitat-sim.git
RUN /bin/bash -c ". activate habitat; cd habitat-sim; git checkout 62bc01be8cbc148eb099e96fa0b123e16b21f95f; python setup.py install --headless"

# Install challenge specific habitat-api
RUN git clone https://github.com/facebookresearch/habitat-api.git
RUN /bin/bash -c ". activate habitat; cd habitat-api; git checkout 0985c6ffd17557150488d238d79574c60612faa9; pip install -e ."

# Silence habitat-sim logs
ENV GLOG_minloglevel=2
ENV MAGNUM_LOG="quiet"
