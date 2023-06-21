# DeepLearnLab
A learning hub for Machine Learning and Deep Learning available through this GitHub repository and (https://boosungkim.com/dllab)[https://boosungkim.com/dllab].

This site contains an accumulation of Pytorch implementations of popular papers and concepts. The project idea was inspired by the popular "Annotated Pytorch Paper Implementations." So far, several milestone Computer Vision models have been implemented. Some future goals include, but are not limited to:

- Implement milestone papers from several other domains, like NLP.
- Include Tensorflow implementations.
- Add experimentation functionality where the users can try out the models with different hyperparameters.
- Implement modern state-of-the-art papers.

The model code implementations were completed by Boosung Kim. The website is based off of (Moonwalk)[https://github.com/abhinavs/moonwalk].


## Prerequisites
Made using:
- Python 3.11.3
- PyTorch 2.0.1
- Torchinfo 1.8.0


## Current List
| Model                                                 | Best Training Accuracy    | Best Test Accuracy          | Notes          |
| -----------------                                     | ----------------------      | ----------------------      | ----------------------      |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 93.4%                       | 88.2%                       | [Blog Post](https://boosungkim.com/blog/2023/first-paper-implementation/). Included BatchNorm to every layer. Removed two final FC |
| [ResNet](https://arxiv.org/pdf/1512.03385)              | 94.7%                       | 88.5%                       |  |
| [DenseNet](https://arxiv.org/abs/1608.06993)              | 96.6%                       | 88.7%                       |  |
| [SE-ResNet](https://arxiv.org/abs/1709.01507)              | 95.9%                       | 88.6%                       |  |