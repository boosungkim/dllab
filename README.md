# Machine Learning Papers Implementations
Implementations of Machine learning papers by Boosung Kim. 

The models are made from scratch in PyTorch with no tutorials. The dataset used is CIFAR10.

## Prerequisites
Made using:
- Python 3.11.3
- PyTorch 2.0.1


## Accuracy
| Model                                                 | Best Validation Accuracy    | Best Test Accuracy          | Notes          |
| -----------------                                     | ----------------------      | ----------------------      | ----------------------      |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 88.7%                       | 88.2%                       | Included BatchNorm to every layer. Removed two final FC |
