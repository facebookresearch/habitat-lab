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
ADD cmake-3.13.3-Linux-x86_64.sh /cmake-3.13.3-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.13.3-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

# Conda environment
RUN conda create -n habitat python=3.6

# Setup habtiat-sim
RUN git clone https://github.com/facebookresearch/habitat-sim.git
RUN /bin/bash -c ". activate habitat; cd habitat-sim; python setup.py install --headless"

# Install habitat-api
RUN git clone https://github.com/facebookresearch/habitat-api.git
RUN /bin/bash -c ". activate habitat; cd habitat-api; pip install -e ."

# Download test data
RUN wget http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip
RUN unzip habitat-test-scenes.zip -d habitat-api/
RUN rm habitat-test-scenes.zip

# Silence habitat-sim logs
ENV GLOG_minloglevel=2
ENV MAGNUM_LOG="quiet"

