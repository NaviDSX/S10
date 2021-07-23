# EVA6 Session 10 (Forked from EVA5 Session 12 my earlier repo)


## Questions in S10-Assignment-Solution:

### 1. Share the final 4-5 log data showing your validation/test accuracy (don't redirect to any page, you'll get 0).
``` python
Epoch 47:
602/602 [========] - 148s 246ms/step - loss: 0.9772 - accuracy: 62.8079
Validation set: Average loss: 0.0143, Accuracy: 55.31%

Epoch 48:
602/602 [========] - 148s 246ms/step - loss: 0.9740 - accuracy: 63.0871
Validation set: Average loss: 0.0143, Accuracy: 55.31%

Epoch 49:
602/602 [========] - 148s 247ms/step - loss: 0.9743 - accuracy: 63.3537
Validation set: Average loss: 0.0143, Accuracy: 55.31%

Epoch 50:
602/602 [========] - 149s 247ms/step - loss: 0.9750 - accuracy: 63.6093
Validation set: Average loss: 0.0143, Accuracy: 55.38%
```
### 2.Describe the data augmentation strategy that you used in points. Then copy page the augmentation/transformation code here (don't redirect to any page, you'll get 0).
The following data augmentation techniques were applied to the dataset during training:
- Horizontal Flip
- Vertical Flip
- Random Rotate
- CutOut

``` python
dataset = TinyImageNet(
    train_batch_size=128,
    val_batch_size=128,
    cuda=cuda,
    num_workers=16,
    horizontal_flip_prob=0.3,
    vertical_flip_prob=0.1,
    rotate_degree=10,
    cutout_prob=0.3,
    cutout_dim=(16, 16),
)


import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensor


class Transformations:
    """Wrapper class to pass on albumentaions transforms into PyTorch."""

    def __init__(
        self, padding=(0, 0), crop=(0, 0), horizontal_flip_prob=0.0,
        vertical_flip_prob=0.0, gaussian_blur_prob=0.0, rotate_degree=0.0,
        cutout_prob=0.0, cutout_dim=(8, 8), mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5), train=True
    ):
        """Create data transformation pipeline.
        
        Args:
            padding (tuple, optional): Pad the image if the image size is less
                than the specified dimensions (height, width). (default= (0, 0))
            crop (tuple, optional): Randomly crop the image with the specified
                dimensions (height, width). (default: (0, 0))
            horizontal_flip_prob (float, optional): Probability of an image
                being horizontally flipped. (default: 0)
            vertical_flip_prob (float, optional): Probability of an image
                being vertically flipped. (default: 0)
            rotate_prob (float, optional): Probability of an image being
                rotated. (default: 0)
            rotate_degree (float, optional): Angle of rotation for image
                augmentation. (default: 0)
            cutout_prob (float, optional): Probability that cutout will be
                performed. (default: 0)
            cutout_dim (tuple, optional): Dimensions of the cutout box (height, width).
                (default: (8, 8))
            mean (float or tuple, optional): Dataset mean. (default: 0.5 for each channel)
            std (float or tuple, optional): Dataset standard deviation. (default: 0.5 for each channel)
        """
        transforms_list = []

        if train:
            if sum(padding) > 0:
                transforms_list += [A.PadIfNeeded(
                    min_height=padding[0], min_width=padding[1], always_apply=True
                )]
            if sum(crop) > 0:
                transforms_list += [A.RandomCrop(crop[0], crop[1], always_apply=True)]
            if horizontal_flip_prob > 0:  # Horizontal Flip
                transforms_list += [A.HorizontalFlip(p=horizontal_flip_prob)]
            if vertical_flip_prob > 0:  # Vertical Flip
                transforms_list += [A.VerticalFlip(p=vertical_flip_prob)]
            if gaussian_blur_prob > 0:  # Patch Gaussian Augmentation
                transforms_list += [A.GaussianBlur(p=gaussian_blur_prob)]
            if rotate_degree > 0:  # Rotate image
                transforms_list += [A.Rotate(limit=rotate_degree)]
            if cutout_prob > 0:  # CutOut
                if isinstance(mean, float):
                    fill_value = mean * 255.0
                else:
                    fill_value = tuple([x * 255.0 for x in mean])
                transforms_list += [A.CoarseDropout(
                    p=cutout_prob, max_holes=1, fill_value=fill_value,
                    max_height=cutout_dim[0], max_width=cutout_dim[1]
                )]
        
        transforms_list += [
            # normalize the data with mean and standard deviation to keep values in range [-1, 1]
            # since there are 3 channels for each image,
            # we have to specify mean and std for each channel
            A.Normalize(mean=mean, std=std, always_apply=True),
            
            # convert the data to torch.FloatTensor
            # with values within the range [0.0 ,1.0]
            ToTensor()
        ]

        self.transform = A.Compose(transforms_list)
    
    def __call__(self, image):
        """Process and image through the data transformation pipeline.
        Args:
            image: Image to process.
        
        Returns:
            Transformed image.
        """
        if not isinstance(image, np.ndarray):
            image = np.array(image)
            
        image = self.transform(image=image)['image']

        if len(image.size()) == 2:
            image = torch.unsqueeze(image, 0)
        return image


def data_loader(data, shuffle=True, batch_size=1, num_workers=1, cuda=False):
    """Create data loader
    Args:
        data (torchvision.datasets): Downloaded dataset.
        shuffle (bool, optional): If True, shuffle the dataset. 
            (default: True)
        batch_size (int, optional): Number of images to considered
            in each batch. (default: 1)
        num_workers (int, optional): How many subprocesses to use
            for data loading. (default: 1)
        cuda (bool, optional): True is GPU is available. (default: False)
    
    Returns:
        DataLoader instance.
    """

    loader_args = {
        'shuffle': shuffle,
        'batch_size': batch_size
    }

    # If GPU exists
    if cuda:
        loader_args['num_workers'] = num_workers
        loader_args['pin_memory'] = True
    
    return torch.utils.data.DataLoader(data, **loader_args)


class InfiniteDataLoader:
    """Create infinite loop in a data loader.
    Args:
        data_loader (torch.utils.data.DataLoader): DataLoader object.
        auto_reset (bool, optional): Create an infinite loop data loader.
            (default: True)
    """

    def __init__(self, data_loader, auto_reset=True):
        self.data_loader = data_loader
        self.auto_reset = auto_reset
        self._iterator = iter(data_loader)

    def __next__(self):
        # Get a new set of inputs and labels
        try:
            data, target = next(self._iterator)
        except StopIteration:
            if not self.auto_reset:
                raise
            self._iterator = iter(self.data_loader)
            data, target = next(self._iterator)

        return data, target

    def get_batch(self):
        return next(self)
 ``` 
      
### 3.Share the link to your repo where you are storing your model.py file (cannot be the same as the repo for this assignment submission).
[model](https://github.com/NaviDSX/S10/blob/main/tensornet/model/base_model.py)

### 4.Share the link to this assignment's repo where we can find the code for TinyImageNet Resnet and the code for calculation of K-means (these can be different if you want).
 - [S12_Part1.ipynb](S12_Part1.ipynb)
 - [S12_Part2.ipynb](S12_Part2.ipynb)
 
### 5.Share the link to README for k-means assignment.
 [S12_Part2.ipynb](S12_Part2.ipynb)
 
>JSON File Structure
>The annotation file will be of COCO format and have the following structure
>
>images
>- id: Image id
>- width: Width of the image
>- height: Height of the image
>- filename: Image file name
>- license: License id for the image
>- date_captured: Date of capture of the image
>- annotations
>- id: Annotation id
>- image_id: Id of the image the annotation is associated with
>- category_id: Id of the class the annotation belongs to
>- segmentation: (x, y) coordinates of the four corners of the bounding box
>- area: Area of the bounding box
>- bbox: (x, y) coordinate of the top-left corner and width and height of the bounding box
>- iscrowd: If the image has a crowd of objects denoted by this annotation

### 6.Share the image showing your BBOXes based on your calculations for all the Ks, and describe them.
[Shown in Notebook](https://github.com/NaviDSX/S10/blob/main/S12_Part2.ipynb)


PS: Note: Tensornet library was refactored from the best assignments after due date.



# Previous Readme

## Questions in S12-Assignment-Solution:

### What is your final accuracy? 
55.42%

### Share the Github link to your ResNet-Tiny-ImageNet code. All the logs must be visible.
 [S12_Part1.ipynb](S12_Part1.ipynb)
 
### Describe the contents of the JSON file in detail. You need to explain each element in detail. 
 [JSON Explained](https://github.com/NaviDSX/S10/blob/main/JSON_Explained.md)
 
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


