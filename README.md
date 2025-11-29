# Denoising-Diffusion-Probabilistic-Model
A lightweight PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) trained on the MNIST dataset.

This repository contains a lightweight implementation of a Denoising Diffusion Probabilistic Model (DDPM) trained on the MNIST dataset. It demonstrates the core principles of generative AI "from scratch," including:
Forward Diffusion Process: A linear noise scheduler that gradually corrupts data.
Reverse Denoising: A custom convolutional neural network (CNN) trained to predict and remove noise at specific timesteps.
Sampling: Generating new handwritten digits from pure Gaussian noise.
