# e4040-2021Spring-project

# Unsupervised Representation Learning Using Deep Convolutional Generative Adversarial Networks

## E4040 Group Project made by:
## Team Name UGAN
Shambhavi Roy (sr3767)

Simran Tiwari (st3400)

Saravanan Govindarajan (sg3896)


## Introduction

The paper by Radford et al.[1] explores the problem of learning image features from large datasets using Generative Adversarial Networks (GANs) instead of deep Convolutional Neural Networks (CNNs). These learned image features can then be used for image classification tasks. This paper also mentions the problem of unstable training of GANs and obtaining nonsensical outputs. To mitigate this, they mention certain architectural constraints for GANs for successful training on large image datasets. This proposed architecture is called Deep Convolutional Generative Adversarial Networks (DCGANs).
Further, this paper shows the use of the trained discriminator model of the GAN for image classification and compares the results with other common image classification algorithms. Finally, this paper presents the concept of the learned generator model having vector arithmetic properties that can be used to understand the quality and properties of the generated image samples.


Our project is based on this paper. Though this paper mentions the subsequent use of the learned feature representations for image classification tasks and further analysis, our project focuses  on training this architecture on multiple datasets to obtain the learned features and visualizing them. 


## Code Organization

The code in this project is organized in multiple Jupyter notebooks and companion .py files. The root directory also consists of other folders with the saved models, screenshots etc. 



## Jupyter notebooks used

DCGAN_CIFAR10_NOBN.ipynb is used to train the DCGAN architecture on the CIFAR-10 dataset [2]

DCGAN_CelebA_NOBN.ipynb is used to train the DCGAN architecture on the CelebA dataset [3]

DCGAN_SVHN_NOBN.ipynb is used to train the DCGAN architecture on the SVHN dataset [4]

DCGAN_LSUN_NOBN.ipynb is used to train the DCGAN architecture on the LSUN dataset [10]

Guided_Backpropagation_LSUN.ipynb shows the application of Guided Backpropagation to the discriminator trained using the LSUN dataset

GradCam_SVHN.ipynb shows the visualization of learned filters using GradCAM [11]


## Supplementary Materials
Project website:  https://ecbme4040.github.io/e4040-2021Spring-Project-UGAN-sr3767-st3400-sg3896/

Google drive folder: https://drive.google.com/drive/folders/1DQ_by-k33v1k-8cY3zsSSCJC30lUg-8q?usp=sharing


## References:

[1] Radford et al.  “Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks”, 	arXiv:1511.06434 [Online] https://arxiv.org/pdf/1511.06434.pdf

[2] CIFAR-10 dataset: Alex Krizhevsky, Learning Multiple Layers of Features from Tiny Images, 2009 [Online] https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

[3] CelebA dataset: Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou, Deep Learning Face Attributes in the Wild, Proceedings of International Conference on Computer Vision (ICCV), December 2015.

[4] SVHN dataset: Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, Andrew Y. Ng Reading Digits in Natural Images with Unsupervised Feature Learning NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011.
[Online] http://ufldl.stanford.edu/housenumbers/

[5] Fisher Yu, Ari Seff, Yinda Zhang, Shuran Song, Thomas Funkhouser and Jianxiong Xiao, LSUN: Construction of a Large-scale Image Dataset using Deep Learning with Humans in the Loop, arXiv:1506.03365 [cs.CV], 10 Jun 2015

[6] Deep Convolutional Generative Adversarial Network [Online] https://www.tensorflow.org/tutorials/generative/dcgan

[7] Having fun with Deep Convolutional GANs[Online] https://naokishibuya.medium.com/having-fun-with-deep-convolutional-gans-f4f8393686ed

[8] Deep Convolutional GAN (DCGAN) with SVHN
[Online] https://github.com/naokishibuya/deep-learning/blob/master/python/dcgan_svhn.ipynb

[9] Goodfellow, NIPS 2016 Tutorial: Generative Adversarial Networks arXiv:1701.00160v4
[Online]https://arxiv.org/pdf/1701.00160.pdf

[10] 20% sample of LSUN dataset https://www.kaggle.com/jhoward/lsun_bedroom

[11] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE International Conference on Computer Vision, pages 618–626, 2017.



## Directory Tree Structure
```
.
├── .gitignore
├── .gitignore.orig
├── DCGAN-CIFAR10-NOBN.ipynb
├── DCGAN_CelebA_NOBN.ipynb
├── DCGAN_LSUN_NOBN.ipynb
├── DCGAN_SVHN_NOBN.ipynb
├── E4040.2021Spring.UGAN.report.sr3767.st3400.sg3896.pdf
├── GradCam_SVHN.ipynb
├── Guided_Backpropagation_LSUN.ipynb
├── README.md
├── UGAN Introduction Notebook.ipynb
├── Utils_Demo.ipynb
├── datasets
│   ├── download_celeba_dataset.sh
│   ├── download_lsun_dataset.sh
│   └── download_svhn_dataset.sh
├── dcgan-lsun-nobn.ipynb
├── model
│   ├── __pycache__
│   │   ├── dcgan.cpython-37.pyc
│   │   ├── dcgan_cifar.cpython-37.pyc
│   │   ├── dcgan_lsun.cpython-37.pyc
│   │   ├── dcgan_net.cpython-37.pyc
│   │   └── modified_dcgan.cpython-37.pyc
│   ├── dcgan.py
│   ├── dcgan_cifar.py
│   ├── dcgan_lsun.py
│   ├── dcgan_net.py
│   └── modified_dcgan.py
└── utils
    ├── __pycache__
    │   ├── dataset_utils.cpython-37.pyc
    │   ├── image_utils.cpython-37.pyc
    │   ├── model_utils.cpython-37.pyc
    │   └── visualizers_utils.cpython-37.pyc
    ├── dataset_utils.py
    ├── image_utils.py
    ├── model_utils.py
    └── visualizers_utils.py

5 directories, 34 files
```