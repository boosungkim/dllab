# Milestone CNN Models Implementations
Implementations of popular Convolutional Neural Network models by Boosung Kim. 

The models are made from scratch in PyTorch with no tutorials. The dataset used is CIFAR10.

## Prerequisites
Made using:
- Python 3.11.3
- PyTorch 2.0.1
- Torchinfo 1.8.0

## Data Augmentation


## Accuracy
| Model                                                 | Best Training Accuracy    | Best Test Accuracy          | Notes          |
| -----------------                                     | ----------------------      | ----------------------      | ----------------------      |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 93.4%                       | 88.2%                       | [Blog Post](https://boosungkim.com/blog/2023/first-paper-implementation/). Included BatchNorm to every layer. Removed two final FC |
| [ResNet](https://arxiv.org/pdf/1512.03385)              | 94.7%                       | 88.5%                       |  |
| [DenseNet](https://arxiv.org/abs/1608.06993)              | 96.6%                       | 88.7%                       |  |
| [SE-ResNet](https://arxiv.org/abs/1709.01507)              | 95.9%                       | 88.6%                       |  |