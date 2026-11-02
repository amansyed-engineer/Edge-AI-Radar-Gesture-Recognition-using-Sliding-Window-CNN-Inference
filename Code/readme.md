# Code Documentation

This folder contains the implementation of the radar-based kick gesture recognition system.

## Overview
The code implements an end-to-end pipeline consisting of:
- Radar signal preprocessing
- Temporal segmentation
- Fixed-length window extraction
- CNN-based classification
- Person-wise evaluation

## Main Components
- Preprocessing: bandpass filtering and amplitude normalization
- Segmentation methods:
  - Energy-based segmentation
  - Gradient-based segmentation
  - Doppler (STFT)-based segmentation
- Classification: 1D Convolutional Neural Network (CNN)

## Execution
All experiments are executed using the Jupyter notebook provided in this folder.

Required Python libraries are listed in the `requirements.txt` file at the repository root.
