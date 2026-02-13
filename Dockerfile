FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NCCL_ASYNC_ERROR_HANDLING=1 \
    NCCL_DEBUG=INFO

# --------------------------------------------------
# System packages
# --------------------------------------------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    curl \
    ca-certificates \
    build-essential \
    vim \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# Python tooling
# --------------------------------------------------
RUN pip3 install --upgrade pip setuptools wheel

# --------------------------------------------------
# PyTorch nightly (CUDA 12.9)
# --------------------------------------------------
RUN pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu129

# --------------------------------------------------
# Ray + ML stack
# --------------------------------------------------
RUN pip install \
    ray[train,tune,air]==2.53.0 \
    mlflow \
    transformers \
    accelerate \
    datasets \
    safetensors \
    einops \
    pyyaml

WORKDIR /code

# --------------------------------------------------
# Default: interactive shell
# --------------------------------------------------
CMD ["/bin/bash"]
