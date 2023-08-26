# Siamese NN implementation
Pytorch Siamese NN based on ResNet/RegNet 
with some tweaks to work with dataset crawled from internet.<br/>
It compares two images (mostly persons photos) to classify them as similar or not (the same photo or not).

## Dataset
Labeling done by Active learning:
1) First, created manually 100 examples
2) Train the model (overfiting is okay) for the beginning
3) Use model to get predictions  on 1000+ more examples
4) Take 100 with the highest error and tune the model
5) Repeat till the dataset is ready

## Which images are similar?
__These images should be considered as similar__, regardless images have diff size, color, 
background.
![img.png](assets/img_similar.png)

__These images should be considered as different__
![img.png](assets/img_diff.png)

It means, the NN should pay attention to the main object on the photo, and should disregard
 the background. 

The goal to implement the NN, which is able to detect the difference.

# Phase 2
- [ ] add data aug
- [ ] track f1 score
- [ ] Optimize model to work on CPU
- [ ] Prepare model for inference
