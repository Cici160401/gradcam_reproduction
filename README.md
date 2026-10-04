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

I used a ResNet50 model pretrained on ImageNet. The model was used only for inference and was not fine-tuned or retrained. The input image was preprocessed using the transformations associated with the pretrained weights. After preprocessing, the image tensor had a shape of `[3, 224, 224]`. `unsqueeze(0)` was used to add the batch dimension to the input tensor because ResNet50 expects its input in batched form, resulting in a shape of `[1, 3, 224, 224]`.

![alt text](image-1.png)

### 2. Target convolutional 

I searched for one of the last convolutional layers of ResNet50 because deep convolutional layers keep spatial information  while they capture high-level semantic carachteristics. I selected model.layer4[2].conv3, which corresponds to the last convolutional layer of the final Bottleneck in layer4. The ouput is [1, 2048, 7, 7], where 2048 are the amount of total feature maps and 7x7 represents the spatial resolution of each feature map.

### 3. Forward pass and target class

I performed a forward pass, obtaining an output with shape [1, 1000], where 1000 corresponds to the number of classes in ImageNet. I used \(c\) to represent the index of the target class and \(y^c\) as the score assigned to that class by the model. Since Grad-CAM is class-discriminative, \(y^c\) is selected so that the gradients can be computed specifically with respect to the score of class \(c\). Following the Grad-CAM formulation, \(y^c\) corresponds to the class score before the softmax operation.

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