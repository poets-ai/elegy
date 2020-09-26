FROM docker.io/nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes -y && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.8-dev \
    python3.8-distutils \
    curl \
    && curl -Lk "https://bootstrap.pypa.io/get-pip.py" | python3.8 && \
    rm -rf /var/lib/apt/lists/*

ENV BASE_URL="https://storage.googleapis.com/jax-releases"
RUN python3.8 -m pip install --upgrade $BASE_URL/cuda102/jaxlib-0.1.51-cp38-none-manylinux2010_x86_64.whl
RUN python3.8 -m pip install --upgrade jax elegy dataget matplotlib typer
ENV XLA_PYTHON_CLIENT_ALLOCATOR="platform"

COPY examples/ /usr/src/app/
