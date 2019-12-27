# Supported datasets

- CamVid
- CityScapes

Note: When referring to the number of classes, the void/unlabeled class is excluded.

## CamVid Dataset

The Cambridge-driving Labeled Video Database (CamVid) is a collection of over ten minutes of high-quality 30Hz footage with object class semantic labels at 1Hz and in part, 15Hz. Each pixel is associated with one of 32 classes.

The CamVid dataset supported here is a 12 class version developed by the authors of SegNet. [Download link here](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid). For actual training, an 11 class version is used - the "road marking" class is combined with the "road" class.

More detailed information about the CamVid dataset can be found [here](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) and on the [SegNet GitHub repository](https://github.com/alexgkendall/SegNet-Tutorial).

## Cityscapes

Cityscapes is a set of stereo video sequences recorded in streets from 50 different cities with 34 different classes. There are 5000 images with fine annotations and 20000 images coarsely annotated.

The version supported here is the finely annotated one with 19 classes.

For more detailed information see the official [website](https://www.cityscapes-dataset.com/) and [repository](https://github.com/mcordts/cityscapesScripts).

The dataset can be downloaded from https://www.cityscapes-dataset.com/downloads/. At this time, a registration is required to download the data.