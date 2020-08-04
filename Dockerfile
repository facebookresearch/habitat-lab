FROM nvidia/opengl:1.0-glvnd-devel-ubuntu18.04 as opengl
FROM nvidia/cuda:10.1-devel-ubuntu18.04 as cudagl

RUN apt-get update && apt-get install -y --no-install-recommends \
    	libglvnd0 \
        libgl1 \
        libglx0 \
        libegl1 \
        libgles2 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=opengl \
    /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    /usr/share/glvnd/egl_vendor.d/10_nvidia.json
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compat32,utility

# Base image
FROM cudagl

# Setup basic packages
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libegl1-mesa-dev \
        libglfw3-dev \
        libglm-dev \
        libjpeg-dev \
        libomp-dev \
        libpng-dev \
        libsm6 \
        libx11-dev \
        pkg-config \
        unzip \
        vim \
        wget \
        zip \
    && rm -rf /var/lib/apt/lists/*

# Install conda
RUN curl -o ~/miniconda.sh \
        -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda install \
        ipython \
        mkl \
        mkl-include \
        numpy \
        pyyaml \
        scipy \
    && conda clean -ya

# Conda environment
RUN conda create -n habitat
WORKDIR /opt
SHELL ["/bin/bash", "-c"]

# Setup habitat-sim
RUN git clone --branch master \
        https://github.com/facebookresearch/habitat-sim.git
RUN source activate habitat && \
    cd habitat-sim && \
    pip install -r requirements.txt && \
    python setup.py install --headless

# Install challenge specific habitat-api
COPY ./ ./habitat-api
RUN source activate habitat && \
    cd habitat-api && \
    pip install -r requirements.txt && \
    pip install -e .

# Silence habitat-sim logs
ENV GLOG_minloglevel=2
ENV MAGNUM_LOG="quiet"

# setup entrypoint
COPY ./entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]
SHELL ["/bin/sh", "-c"]
CMD ["bash"]

# WORKDIR /opt/habitat-api
# RUN wget http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip && \
#     unzip habitat-test-scenes.zip
