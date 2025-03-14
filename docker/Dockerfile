# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:30

RUN apt-get update && apt-get install -y \
    git \
    zip \
    vim

# Required by pyvista PIP package
RUN apt-get install -y libglfw3-dev libgles2-mesa-dev

WORKDIR /app/
RUN wget https://developer.download.nvidia.com/devtools/nsight-systems/NsightSystems-linux-cli-public-2023.4.1.97-3355750.deb
RUN apt install ./NsightSystems-linux-cli-public-2023.4.1.97-3355750.deb

RUN pip uninstall -y torch

COPY dev-requirements.txt /app/dev-requirements.txt
RUN pip install -r dev-requirements.txt