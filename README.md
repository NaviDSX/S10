# Session 12


## Questions in S12-Assignment-Solution:

### What is your final accuracy? 
55.42%

### Share the Github link to your ResNet-Tiny-ImageNet code. All the logs must be visible.
 [S12_Part1.ipynb](S12_Part1.ipynb)
 
### Describe the contents of the JSON file in detail. You need to explain each element in detail. 
 [JSON File Explained](JSON.md)
 
### Share the link to your Github file where you have calculated the best K clusters for your dataset. 
 [S12_Part2.ipynb](S12_Part2.ipynb)
 
### Share the link to your hardhat, vest, mask and boots Images Folder on GitHub
 [Images](images)
 
### Share the link to your JSON file on GitHub
 [JSON File](S12_New.json)


## Part 1 - ResNet18 on Tiny ImageNet Dataset


The model reaches a maximum accuracy of **55.42%** on Tiny-ImageNet using **ResNet 18** model.

### Parameters and Hyperparameters

- Loss Function: Cross Entropy Loss (combination of `nn.LogSoftmax` and `nn.NLLLoss`)
- Optimizer: SGD
  - Momentum: 0.9
  - Learning Rate: 0.01
- Reduce LR on Plateau
  - Patience: 2
  - Factor: 0.1
  - Min LR: 1e-6
- Epochs: 50
- Batch Size: 128

### Data Augmentation

The following data augmentation techniques were applied to the dataset during training:

- Horizontal Flip
- Vertical Flip
- Random Rotate
- CutOut


## Part 2 - Finding anchor boxes and the clusters.

Finding anchor boxes using **K-Means Clustering ALgorithm**. The dataset was annotated and exported in _COCO JSON Format_.


