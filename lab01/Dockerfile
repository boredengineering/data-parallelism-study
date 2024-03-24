# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# 
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
###############################################################################

FROM nvcr.io/nvidia/pytorch:22.08-py3

RUN pip install ipympl==0.9.2 jupyterlab==3.4.8 matplotlib==3.6.1 --upgrade

# Load saved JupyterLab workspace 
COPY jupyter/ /root/.jupyter/

# Use bash, make nicer
ENV SHELL /bin/bash
COPY jupyter/bashrc /root/.bashrc

# Create working directory to add repo.
WORKDIR /dli

# Load contents into student working directory.
ADD . .

# Create working directory for students.
WORKDIR /dli/task

ENV SHELL=/bin/bash

ADD entrypoint.sh /usr/local/bin
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]