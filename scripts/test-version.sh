#!/bin/bash
# run-test - A script to run tests pypeln in a container
# can recive an optional declaring python version
set -e

kPYTHON_VERSIONS='^[3]\.[0-9]{1,2}$'
kDEFAULT_VERSION=3.8


container_runner () {
    if [[ -z "$1" ]]; then
        py_version="$kDEFAULT_VERSION"
    else
        py_version=$1
    fi

    if hash podman 2>/dev/null; then
        podman build --build-arg PY_VERSION="$py_version" -t elegy .
        podman run -it --rm  -v "$(pwd)":/usr/src/app:Z elegy:latest
    else
        docker build --build-arg PY_VERSION="$py_version" -t elegy .
        docker run -it --rm  -v "$(pwd)":/usr/src/app:Z elegy:latest
    fi
}

if [[ $1 =~ $kPYTHON_VERSIONS ]] || [[ -z "$1" ]]; then
    container_runner "$1"
else
    echo "Check python version"
fi