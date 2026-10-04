# Grad-CAM from Scratch: Paper Reproduction

## Overview

This repository is a reproduction of the paper Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. The Grad-CAM method was implemented from scratch to obtain the class-discriminative localization map built by weighting feature maps according to their importance for the target class score. The method was implemented step by step from the equations stated in the paper. I used the pre-trained ResNet50 with ImageNet to understand how Grad-CAM works internally, and the implementation was validated against the pytorch-grad-cam reference implementation.

## Paper

> Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017).  
> [*Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*](https://arxiv.org/abs/1610.02391).  
> Proceedings of the IEEE International Conference on Computer Vision (ICCV).

CNNs achieve strong performance but their predictions are difficult to interpret, this makes failures/biases harder to understand, so Grad-CAM provides spatial information about which regions contribute to a particular class.

## Objective of this reproduction

## Method
### 1. Model and input



### 2. Target convolutional layer
### 3. Forward pass and target class
### 4. Capturing feature maps
### 5. Computing gradients
### 6. Computing channel importance weights
### 7. Building the Grad-CAM localization map
### 8. Visualization

## Results

## Validation
### Reference implementation
### Output comparison
### Raw Grad-CAM validation

## Experiments
[pendiente]

## Repository structure

## Installation and usage

## Key takeaways

## References