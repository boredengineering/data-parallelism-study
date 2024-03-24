# Data Parallelism Study

Study material, notes and complementary material on Distributed Data Parallel (DDP) for PyTorch based on the Nvidia DLI lab.

**Author:** 
> - Renan Monteiro Barbosa

# Introduction

The study focus on accelerating the training of a simple Torchvision model using DDP. The same process can be replicated using TorchText https://pytorch.org/text/stable/index.html.

In the Future must explore adding Megatron-Deepspeed DDP methods as well model-parallelism. Often is common during the development to use mustiple strategies therefore engineering planning done ahead can avoid hundreds of hours in refactoring.


# TODO

- Must configure Docker Compose and the Entrypoint.sh scripts.
- Develop a simple Terraform and Ansible automation to facilitate Cloud deployment.
- GPU cluster configuration is missing.