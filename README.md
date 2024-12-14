# EfficientNetV2 CIFAR-10 and CIFAR-100 Testing

This repository includes notebooks and code for testing EfficientNetV2 models on CIFAR-10 and CIFAR-100 datasets. It also introduces an alternative upsampling-based approach explored in the [`adjustment.ipynb`](adjustment.ipynb) notebook.

## Folders and Weights

To use the pretrained weights, create a `Weights` directory in the root folder and download the weights from the following [Google Drive link](https://drive.google.com/drive/folders/1A_0bhfMhy23ppDSK7CfcoqENe4IeR6QW?usp=sharing).

### Pretrained Weights
The pretrained weights correspond to three versions of EfficientNetV2:
- **Small**: `efficientnet_v2_s`
- **Medium**: `efficientnet_v2_m`
- **Large**: `efficientnet_v2_l`

## Notebooks Overview

### 1. [`cifar_10_testing.ipynb`](cifar_10_testing.ipynb) and [`cifar_100_testing.ipynb`](cifar_100_testing.ipynb)

These notebooks test the accuracy of EfficientNetV2 models on the CIFAR-10 and CIFAR-100 datasets using the pretrained weights. The testing results are summarized below:

| Dataset   | Model Size | Accuracy (%) |
|-----------|------------|--------------|
| CIFAR-10  | Small      | 98.46%       |
|           | Medium     | 98.91%       |
|           | Large      | 98.80%       |
| CIFAR-100 | Small      | 90.96%       |
|           | Medium     | 91.53%       |
|           | Large      | 91.88%       |

EfficientNetV2 models demonstrate excellent performance across both datasets, with Medium and Large variants achieving the best accuracies.

### 2. [`adjustment.ipynb`](adjustment.ipynb)

This notebook experiments with an alternative approach to resizing input images. Instead of using standard transformations like resizing and center cropping, an **Upsample Module** is introduced. The module uses bilinear interpolation combined with convolutional layers and Squeeze-and-Excitation (SE) blocks to upscale CIFAR-10 images (32x32) to 224x224 dimensions before passing them to the EfficientNetV2 model.

#### Results
Using this approach, the model achieved a **Test Loss of 0.5809** and a **Test Accuracy of 96.65%** on the CIFAR-10 dataset. This demonstrates that learning-based resizing methods can offer competitive performance compared to traditional transformations.

### Upsample Module
The `UpsampleModule` combines:
- Bilinear upsampling
- Convolutional layers
- Batch normalization
- ReLU activation
- Squeeze-and-Excitation blocks for channel attention

### Key Classes in the Code
- **`SEBlock`**: Implements Squeeze-and-Excitation for channel-wise attention.
- **`UpsampleModule`**: Upsamples input images using bilinear interpolation and convolutional layers.
- **`EfficientNetV2_FeatureExtraction`**: Loads EfficientNetV2 models with the pretrained weights and integrates the Upsample Module.

## Credits

Most of the EfficientNetV2 implementation is sourced from the [EfficientNetV2 PyTorch repository](https://github.com/hankyul2/EfficientNetV2-pytorch). The weights and models were directly loaded from this repository.
