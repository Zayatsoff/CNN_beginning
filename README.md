## PyTorch implementation of [AlexNet](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) trained on [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
Classifying between 10 different cotegories of 32x32 images.

Differences:
* Optimizer : Adam instead of SGD
#### Train Set Accuracy : 99.82%  
#### Validation Set Accuracy : 79.58%

TODO:
- [ ] Reduce overfitting


You can download the model weights [here](https://anonfiles.com/H3S2BfIex7/checkpoint_pt).

## Simple CNN classifier in TensorFlow trained on a [Malaria Dataset from NLM](https://lhncbc.nlm.nih.gov/LHC-publications/pubs/MalariaDatasets.html)
Parasitized vs uninfected cells
* Conv2D : 32 kernels each 3x3 Act: RELU
* MaxPool2D 2x2
* Conv2D : 64 kernels each 3x3 Act: RELU
* MaxPool2D 2x2
* Conv2D : 64 kernels each 3x3 Act: RELU
* MaxPool2D : 2x2
* Dense : 128 Act: RELU
* Dropout : 0.5
* Optimizer : Adam

### Accuracy : 94%
