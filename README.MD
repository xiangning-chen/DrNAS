# DrNAS

Code accompanying the paper  
***DrNAS: Dirichlet Architecture Search*** [paper](https://arxiv.org/abs/2006.10355)  
Xiangning Chen, Ruochen Wang, Minhao Cheng, Xiaocheng Tang, Cho-Jui Hsieh

This code is based on the implementation of [NAS-Bench-201](https://github.com/D-X-Y/AutoDL-Projects) and [PC-DARTS](https://github.com/yuhuixu1993/PC-DARTS).

## Architecture Search

**Search on NAS-Bench-201 Space: (3 datasets to choose from)**

* Data preparation: Please first download the 201 benchmark file and prepare the api follow [this repository](https://github.com/D-X-Y/NAS-Bench-201).

* ```cd 201-space && python train_search_progressive.py```

**Search on DARTS Space:**

* Data preparation: Please first sample 10% and 2.5% images for earch class as the training and validation set.

* ```CIFAR-10: cd DARTS-space && python train_search.py```

* ```ImageNet: cd DARTS-space && python train_search_imagenet.py```

## Architecture Evaluation

* ```CIFAR-10: cd DARTS-space && python train.py --cutout --auxiliary```

* ```ImageNet: cd DARTS-space && python train_imagenet.py --auxiliary```