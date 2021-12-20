# CNN_Cells
## [Malaria Dataset from NLM](https://lhncbc.nlm.nih.gov/LHC-publications/pubs/MalariaDatasets.html)
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

#### Accuracy : 94%
