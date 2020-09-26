#!/bin/bash
# test-gpu - A script to run tests elegy in a gpu enabled container
set -e


container_runner () {

    if hash podman 2>/dev/null; then
        podman build -f Dockerfile_CUDA -t elegy:latest
        podman run --privileged  -it --rm --security-opt=label=disable -v "$(pwd)":/usr/src/app:Z -e NVIDIA_VISIBLE_DEVICES=all elegy bash
    else
    docker build -f Dockerfile_CUDA -t elegy:latest
    docker run --privileged  -it --rm --security-opt=label=disable -v "$(pwd)":/usr/src/app:Z -e NVIDIA_VISIBLE_DEVICES=all elegy bash
    
    fi
}

container_runner 