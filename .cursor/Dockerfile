# Cloud Agent base image. Cursor checks out the repo separately — do not COPY source.
FROM python:3.11-bookworm

USER root
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        curl \
        ca-certificates \
        gnupg \
        sudo \
        tesseract-ocr \
        tesseract-ocr-ara \
        antiword \
        catdoc \
        libgl1 \
        libglib2.0-0 \
        libmagic1 \
        unrar-free \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# python:3.11-bookworm already ships /usr/local/bin/python3.11 (python3 is the
# same file). `ln -sf "$(command -v python3)" /usr/local/bin/python3.11` fails
# with "are the same file" and aborts every environment Docker build.

RUN if ! id ubuntu >/dev/null 2>&1; then useradd -m -s /bin/bash -u 1000 ubuntu; fi \
    && echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu

USER ubuntu
