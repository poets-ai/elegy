ARG PY_VERSION=3.8
FROM docker.io/python:${PY_VERSION}

RUN pip install jaxlib jax elegy dataget matplotlib typer
WORKDIR /usr/src/app
COPY examples/ /usr/src/app/

