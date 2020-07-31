# DeepCorrect: *Correcting DNN models against Image Distortions*

## Introduction
In recent years, the widespread use of deep neural networks (DNNs) has facilitated great improvements in performance for computer vision tasks like image classification and object recognition. In most realistic computer vision applications, an input image undergoes some form of image distortion such as blur and additive noise during image acquisition or transmission. Deep networks trained on pristine images perform poorly when tested on such distortions. *DeepCorrect* improves the robustness of pre-trained DNN models by training small stacks of convolutional layers with *residual* connections at the output of the most distortion susceptible convolutional filters in a DNN, to correct their filter activations, whilst leaving the rest of the pre-trained DNN filter outputs unchanged. Performance results show that applying *DeepCorrect* models for common vision tasks like image classification (CIFAR-100, ImageNet), object recognition (Caltech-101, Caltech-256) and scene classification (SUN-397), significantly improves the robustness of DNNs against distorted images and outperforms the alternative approach of network fine-tuning.

A complete description of *DeepCorrect* can be found in our journal paper [IEEE Transactions on Image Processing](https://ieeexplore.ieee.org/document/8746775) or in a pre-print on [ArXiv](https://arxiv.org/abs/1705.02406). 


   **2-dimensional t-SNE embedding of baseline AlexNet DNN features for ImageNet object classes :**
   

   ![baseline AlexNet](https://github.com/tsborkar/DeepCorrect_archive/blob/master/eps_fig/Fig3_1.png)




   **2-dimensional t-SNE embedding of DeepCorrect features for ImageNet object classes :**


  ![baseline AlexNet](https://github.com/tsborkar/DeepCorrect_archive/blob/master/eps_fig/Fig15_1.png)


   **Deep neural network architectures :**
   
   Convolution layers in the *DeepCorrect* models, shown in gray with dashed outlines, are nontrainable layers and their weights are kept the same as those of the baseline models trained on undistorted images. 
  


  ![model_arch](https://github.com/tsborkar/DeepCorrect_archive/blob/master/eps_fig/model_fig.png)



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

## *DeepCorrect* performance results

#### Top-1 accuracy for undistorted images

|   Model        |  ImageNet (2012) |    SUN-397  |   Caltech-101  |  Caltech-256   |  CIFAR-100  |
| :-----------:  | :--------------: | :---------: |   :----------: |   :---------:  | :---------: |
|  Baseline      |    0.5694        |   0.3100    |     0.8500     |    0.6200      |   0.7028    |


#### Top-1 accuracy for Gaussian blur affected images, averaged over all levels of distortion


|   Model        |  ImageNet (2012) |    SUN-397  |   Caltech-101  |  Caltech-256   |  CIFAR-100  |
| :-----------:  | :--------------: | :---------: |   :----------: |   :---------:  | :---------: |
|  Baseline      |    0.2305        |   0.1393    |     0.4980     |    0.2971      |   0.2502    |
|  Finetuning    |    0.4596        |   0.2369    |     0.7710     |    0.5167      |   0.5727    |
| *DeepCorrect*  |   **0.5071**     | **0.3049**  |   **0.8371**   |   **0.5883**   | **0.6023**  |


#### Top-1 accuracy for AWGN affected images, averaged over all levels of distortion

|   Model        |  ImageNet (2012) |    SUN-397  |   Caltech-101  |  Caltech-256   |  CIFAR-100  |
| :-----------:  | :--------------: | :---------: |   :----------: |   :---------:  | :---------: |
|  Baseline      |    0.2375        |   0.0859    |     0.3423     |    0.1756      |   0.3147    |
|  Finetuning    |    0.4894        |   0.1617    |     0.7705     |    0.4995      |   0.6451    |
| *DeepCorrect*  |   **0.5092**     | **0.2936**  |   **0.8034**   |   **0.5482**   | **0.6452**  |

#### Trainable parameters 

|   Model        |  ImageNet (2012) |  CIFAR-100  |
| :-----------:  | :--------------: | :---------: |
|  Finetuning    |    60.94 M       |   1.38 M    |
| *DeepCorrect*  |      2.81 M      |     0.89 M  |


#### Accelerating training

   ImageNet validation set accuracy vs. training iterations


  ![train_time](https://github.com/tsborkar/DeepCorrect_archive/blob/master/eps_fig/training_times.png)
  
#### Qualitative evaluation for ImageNet images

   SSIM vs. Filter index


  ![ssim](https://github.com/tsborkar/DeepCorrect_archive/blob/master/eps_fig/ssim-crop.png)
  
   PSNR vs. Filter index

  ![psnr](https://github.com/tsborkar/DeepCorrect_archive/blob/master/eps_fig/psnr-crop.png)


A complete description of the results and the corresponding experimental setup can be found
in the [arXiv tech report](https://arxiv.org/abs/1705.02406).


## Installing *DeepCorrect*

This repo provides a python-based implementation for *DeepCorrect* using the deep learning library Keras and a Theano backend.

**Note**: The current version of code does not support a Tensorflow backend.


#### Prerequisites
- [Python-2.7](https://www.python.org/download/releases/2.7/) (tested with 2.7.12)
- [Theano](http://deeplearning.net/software/theano_versions/0.9.X/)  (tested with 0.9.0)
- [Keras](https://keras.io/) (tested with 1.2.1)
- [h5py](http://www.h5py.org/) 
- [numpy](http://www.numpy.org/)
- [OpenCV](https://github.com/opencv/opencv) (tested with 3.1.0)
- [convnet-keras](https://github.com/heuritech/convnets-keras) (needed only for AlexNet/ ImageNet model)
- MATLAB (needed only for organizing ImageNet image files)

#### Install *DeepCorrect*
Get the *DeepCorrect* source code by cloning the repository :
```
git clone https://github.com/tsborkar/DeepCorrect.git
```

#### Setting up ImageNet (ILSVRC2012) validation and training data
1. Change to *DeepCorrect* source directory: ``` cd DeepCorrect ```
2. Create folders for training and validation data ``` mkdir Training ILSVRC_data ```
3. Download [ILSVRC2012](http://image-net.org/challenges/LSVRC/2012/index) training and validation data files
4. Extract validation set files to ```ILSVRC_data``` folder and training set files to ```Training``` folder respectively.
5. Change to ```misc``` folder in the ```DeepCorrect``` source directory: ``` cd DeepCorrect/misc ```
6. Start MATLAB ```matlab``` and run ```ILSVRC_data_org.m```
```
>> ILSVRC_data_org
```

#### Downloading pre-computed ImageNet models
Due to the large size of ImageNet models, both the finetuned models as well as various trained *DeepCorrect* models need to be downloaded from an external source. To download these models run the bash script ``` get_Imagenet_models.sh``` file from the source ```DeepCorrect``` folder.
```
sh get_Imagenet_models.sh
```

After execution, the ```Imagenet_models``` folder will be populated with finetuned models as well as models of all *DeepCorrect* architectures ( CW, lite, bottleneck), for Gaussian blur and AWGN.

## Runnning *DeepCorrect*

### Correction priority 
To compute correction priorities for pre-trained DNN models, set ```filter_ranked = 0``` in ```cifar_layer_swap.py``` (CIFAR-100) or ```alexnet_imagenet_layer_swap.py``` (ImageNet) and run it.
``` 
python cifar_src/cifar_layer_swap.py
```
or
``` 
python Imagenet_src/alexnet_imagenet_layer_swap.py
```

**Note**: Computing correction priority requires local disk space upto 50GB as feature maps for all layers of the pre-trained DNN need to be stored. Depending on the size of the validation set used, this may take a while.

To use pre-computed correction priorities and plot accuracies for correcting various percentages of filters in a pre-trained DNN's layers, set ```filter_ranked = 1``` in ```cifar_layer_swap.py``` (CIFAR-100) or ```alexnet_imagenet_layer_swap.py``` (ImageNet) and run it.
``` 
python cifar_src/cifar_layer_swap.py
```
or
``` 
python Imagenet_src/alexnet_imagenet_layer_swap.py
```
   **Top-1 accuracy for correcting various percentages of filters**
![ideal_corr](https://github.com/tsborkar/DeepCorrect_archive/blob/master/eps_fig/gen_layer_corr.png)


### Training and testing *DeepCorrect* models

For testing pre-computed *DeepCorrect* models, set ```num_epoch = 0``` in ```cifar_10_100_deepcorr_tr.py``` (CIFAR-100) or ```imagenet_tr_alexnet.py``` and run it.

``` 
python cifar_src/cifar_10_100_deepcorr_tr.py
```
or
``` 
python Imagenet_src/imagenet_tr_alexnet.py
```

For training *DeepCorrect* models, set ```num_epoch > 0``` in ```cifar_10_100_deepcorr_tr.py``` (CIFAR-100) or ```imagenet_tr_alexnet.py``` and run it.

``` 
python cifar_src/cifar_10_100_deepcorr_tr.py
```
or
``` 
python Imagenet_src/imagenet_tr_alexnet.py
```
**Note**: In my experiments, ```num_epoch``` was set to 40 for CIFAR-100 and 50 for ImageNet.








   
 
 









