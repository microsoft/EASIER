# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:22

RUN apt-get update && apt-get install -y \
    git \
    zip \
	vim

WORKDIR /app/
RUN wget https://developer.download.nvidia.com/devtools/nsight-systems/NsightSystems-linux-cli-public-2023.4.1.97-3355750.deb
RUN apt install ./NsightSystems-linux-cli-public-2023.4.1.97-3355750.deb

# WARNING
# mgmetis wrongly specifies Cython a "install time" requirement and mpi4py
# as optional but they should both be "setup time" requirements.
# Therefore we need to install them separatedly before
# `pip install mgmetis` or `pip install -r dev-requirements.txt`
# so that ParMETIS support can be added.
RUN pip install \
	Cython==3.0.11 \
	mpi4py==3.1.5

COPY dev-requirements.txt /app/dev-requirements.txt
RUN pip install -r dev-requirements.txt