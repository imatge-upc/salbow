# Saliency Weighted Convolutional Features for Instance Search

## Abstract
This work explores attention models to weight the contribution of local convolutional representations for the instance search task. We present a retrieval framework based on bags of local convolutional features (BLCF) that benefits from saliency weighting to build an efficient image representation. The use of human visual attention models (saliency) allows significant improvements in retrieval performance without the need to conduct region analysis or spatial verification, and without requiring any feature
fine tuning. We investigate the impact of different saliency models, finding that higher performance on saliency benchmarks does not necessarily equate to improved performance when used in instance search tasks. The proposed approach outperforms the state-of-the-art on the challenging INSTRE benchmark by a large margin, and provides similar performance on the Oxford and Paris benchmarks compared to more complex methods that use off-the-shelf representations.

## Code Instructions
This repo contains python scripts to build Bag of Visual Words based on local CNN features to perform instance search in three different datasets:

* [Instre](ftp://ftp.irisa.fr/local/texmex/corpus/instre/readme.htm), following the evaluation protocol from [Efficient Diffusion on Region Manifolds: Recovering Small Objects with Compact CNN Representations](http://people.rennes.inria.fr/Ahmet.Iscen/diffusion.html).
* [Oxford Buildings](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/).
* [Paris Buildings](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/).

### Prerequisits

* Create a virtual enviroment [how to create a virtual enviroment](http://docs.python-guide.org/en/latest/dev/virtualenvs/).
```
 virtualenv ~/salbow
 source ~/salbow/bin/activate
```
* The code runs with CUDA Version 7.5.18. For python dependencies run:
```
 pip install --upgrade pip
 pip install -r requirements.txt
```

* Then, install the custom [python-vlfeat]((https://github.com/dougalsutherland/vlfeat-ctypes)) library by running:
```
 python install_vlfeat.py
```
* Lastly, modify  *config.py* file to set custom paths:

````
# dataset images are automatically downloaded in:
PATH_DATASET='custom/dataset/path'

# saliency masks for each of the datasets are stored in:
PATH_SALIENCY='custom/saliency/path'

# BLCF models, features and assignment maps are stored in:
PATH_OUTPUT='custom/output/path'
````


### How to run it
```
Usage: python evaluation.py [OPTIONS]

Options:
  --dataset TEXT     Selected dataset for extraction  (availables 'instre' (default), 'oxford', 'paris')
  --layer TEXT       layer from vgg16                 (default 'conv5_1')
  --max_dim INTEGER  Max dimension of images          (default '340')
  --weighting TEXT   Spatial weighting scheme         (availables None (default), 'gaussian', 'l2norm', 'SALGAN')
  --global_search    Flag to apply global search for queries        
  --query_expansion  Flag to apply Average Query Expansion    
```
Example:
```
python evaluation.py --dataset 'instre' --query_expansion --weighting 'SALGAN'

ret:
mAP = 0.697773325515
+QE mAP = 0.757181174096
```
The command above applies saliency weighting from [SalGAN](https://imatge-upc.github.io/saliency-salgan-2017/)
to the assignment maps of Instre. Additionally, query expansion (top 10) images in performed.
Script with [provided data](https://drive.google.com/drive/folders/18NmIcyEIJ8p9GO14rUB3n3wTnx8pezt_) returns:
```

```
