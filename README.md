# DeepCorrect: *Correcting DNN models against Image Distortions*
Created by Tejas Borkar and Lina Karam at Arizona State University ECEE.

## Introduction
In recent years, the widespread use of deep neural networks (DNNs) has facilitated great improvements in performance for computer vision tasks like image classification and object recognition. In most realistic computer vision applications, an input image undergoes some form of image distortion such as blur and additive noise during image acquisition or transmission. Deep networks trained on pristine images perform poorly when tested on such distortions. *DeepCorrect* improves the robustness of pre-trained DNN models by training small stacks of convolutional layers with *residual* connections at the output of the most distortion susceptible convolutional filters in a DNN, to correct their filter activations, whilst leaving the rest of the pre-trained DNN filter outputs unchanged. Performance results show that applying *DeepCorrect* models for common vision tasks like image classification (CIFAR-100, ImageNet), object recognition (Caltech-101, Caltech-256) and scene classification (SUN-397), significantly improves the robustness of DNNs against distorted images and outperforms the alternative approach of network fine-tuning.

A complete description of *DeepCorrect* and corresponding results can be found in an [arXiv tech report](https://arxiv.org/abs/1705.02406).

## Citing *DeepCorrect*
If you use *DeepCorrect* in your research, please consider citing:

```
@article{BorkarK17,
  author    = {Tejas S. Borkar and Lina J. Karam},
  title     = {DeepCorrect: Correcting {DNN} models against Image Distortions},
  journal   = {CoRR},
  volume    = {abs/1705.02406},
  year      = {2017},
  url       = {http://arxiv.org/abs/1705.02406},
  archivePrefix = {arXiv},
  eprint    = {1705.02406},
}

```
## License
*DeepCorrect* is released under the MIT License (refer to the LICENSE file for details).



