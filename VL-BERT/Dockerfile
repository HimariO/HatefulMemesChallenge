FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
RUN mkdir /workspace/
RUN mkdir /workspace/VL-BERT
COPY ./requirements.txt /workspace/VL-BERT
RUN apt-get update \
    && apt-get install -y curl vim
RUN apt install -y python3.7 \
    # && rm /usr/bin/python3 \
    && ln -s python3.7 /usr/bin/python3
RUN apt-get update \
    && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    pkg-config \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    libpq-dev \
    libsm6 libxrender1 libfontconfig1 \
    ffmpeg \
    htop \
    libopenblas-dev \
    libomp-dev \
    tmux \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN cd /workspace/VL-BERT \
    && pip3 install --upgrade pip \
    && pip3 install Cython \
    && pip3 install -f https://download.pytorch.org/whl/torch_stable.html -r requirements.txt
RUN cd /workspace \
    && git clone https://github.com/NVIDIA/apex \
    && cd apex \
    && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get install -y apt-transport-https ca-certificates \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update \
    && apt-get install -y google-cloud-sdk
RUN export GCSFUSE_REPO=gcsfuse-xenial \
    && echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt-get update \ 
    && apt-get install -y gcsfuse
RUN sed -i "1iforce_color_prompt=true" ~/.bashrc
ENV LC_ALL=C.UTF-8
RUN pip3 install spacy-langdetect \
    && python3 -m spacy download en_core_web_lg \
    && pip3 install tensorflow-determinism
RUN pip3 install tensorflow_hub
WORKDIR /workspace
