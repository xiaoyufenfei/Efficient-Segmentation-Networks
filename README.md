## Efficient-Segmentation-Networks
[![python-image]][python-url]
[![pytorch-image]][pytorch-url]

This project aims at providing an easy-to-use, modifiable reference implementation for real-time semantic segmentation models using PyTorch.

<p align="center"><img width="100%" src="docs/image-1.png" /></p>
---

---


#### Model
The project supports these semantic segmentation models as follows:
- [SQNet](https://openreview.net/pdf?id=S1uHiFyyg)
- [LinkNet](https://arxiv.org/pdf/1707.03718.pdf)
- [SegNet](https://arxiv.org/pdf/1511.00561.pdf)
- [UNet](https://arxiv.org/pdf/1505.04597.pdf)
- [ENet](https://arxiv.org/pdf/1606.02147.pdf)
- [ERFNet](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17iv.pdf)
- [EDANet](https://arxiv.org/ftp/arxiv/papers/1809/1809.06323.pdf)
- [ESPNet](https://arxiv.org/pdf/1803.06815v2.pdf)
- [ESPNetv2](https://arxiv.org/pdf/1811.11431.pdf)
- [LEDNet](https://arxiv.org/pdf/1905.02423v3.pdf)
- [ESNet](https://arxiv.org/pdf/1906.09826v1.pdf)
- [FSSNet](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8392426)
- [CGNet](https://arxiv.org/pdf/1811.08201.pdf)
- [DABNet](https://arxiv.org/pdf/1907.11357.pdf)
- [Fast-SCNN](https://arxiv.org/pdf/1902.04502.pdf)
- [ContextNet](https://arxiv.org/pdf/1805.04554.pdf)
- [FPENet](https://arxiv.org/pdf/1909.08599v1.pdf)
- ...

#### Losses

 The project supports these loss functions: 

> 1. Cross Entropy
> 2. Cross Entropy with Label Smooth
> 3. Focal Loss
> 4. Ohem Cross Entropy
> 5. LovaszSoftmax
> 6. ...

#### Optimizers

 The project supports these optimizers: 

> 1. SGD
> 2. Adam 
> 3. AdamW 
> 4. RAdam
> 5. RAdam + Lookahead
> 6. ...

#### Activations

> 1. ReLu
> 2. PReLU
> 3. ReLU6
> 4. Swish
> 5. [Mish](https://github.com/digantamisra98/Mish) : A Self Regularized Non-Monotonic Neural Activation Function
> 6. ...

#### Learning Rate Scheduler

The project supports these LR_Schedulers: 

> 1. Poly decay
> 2. Warm up 
> 3. ...

#### Some useful Tools

> 1. [pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter)
> 2. [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) 
> 3. [Netron](https://github.com/lutzroeder/Netron) : Visualizer for neural network models, On line URL: [Netron](https://lutzroeder.github.io/netron/)
> 4. ...

#### Dataset Setting

This project has been tailored to suit the [Cityscapes](https://www.cityscapes-dataset.com/) and  [Camvid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)  datasets. The folds of your dataset need satisfy the following structures: 

```
|-- dataset
|  |-- camvid
|  |  |-- train
|  |  |-- trainannot
|  |  |-- val
|  |  |-- valannot
|  |  |-- test
|  |  |-- testannot
|  |  |-- ...
|  |-- cityscapes
|  |  |-- leftImg8bit
|  |  |  |-- train
|  |  |  |-- val
|  |  |  |-- test
|  |  |-- gtFine
|  |  |  |-- train
|  |  |  |-- val
|  |  |  |-- test
|  |  |-- ...
```

#### Usage

##### Clone this Repo

```
git clone https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks
cd Efficient-Segmentation-Networks
```

Currently, the code supports [Python 3](https://www.python.org/)

- [PyTorch](https://pytorch.org/) (>=1.1.0)
- torchvision(>=0.3.0)

##### Training

##### Testing

##### Predicting

##### Evaluating

#### Contact

If you think this work useful, please give me a star! And if you find any errors or have any suggestions, please contact me.

**GitHub:** `xiaoyufenfei`
 **Email:** `wangy314159@163.com`





[python-image]: https://img.shields.io/badge/Python-3.x-ff69b4.svg
[python-url]: https://www.python.org/
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.1-2BAF2B.svg
[pytorch-url]: https://pytorch.org/

