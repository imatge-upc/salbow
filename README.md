# Saliency Weighted Convolutional Features for Instance Search

## Abstract
This work explores attention models to weight the contribution of local convolutional representations for the instance search task. We present a retrieval framework based on bags of local convolutional features (BLCF) that benefits from saliency weighting to build an efficient image representation. The use of human visual attention models (saliency) allows significant improvements in retrieval performance without the need to conduct region analysis or spatial verification, and without requiring any feature
fine tuning. We investigate the impact of different saliency models, finding that higher performance on saliency bench- marks does not necessarily equate to improved performance when used in instance search tasks. The proposed approach outperforms the state-of-the-art on the challenging INSTRE benchmark by a large margin, and provides similar performance on the Oxford and Paris benchmarks compared to more complex methods that use off-the-shelf representations.

## Code Instructions
This repo contains python scripts to build Bag of Visual Words based on local CNN features to perform instance search in three different datasets:

* [Instre](ftp://ftp.irisa.fr/local/texmex/corpus/instre/readme.htm), following the evaluation protocol from [Efficient Diffusion on Region Manifolds: Recovering Small Objects with Compact CNN Representations](http://people.rennes.inria.fr/Ahmet.Iscen/diffusion.html).
* [Oxford Buildings](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/).
* [Paris Buildings](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/).

### Prerequisits

* Create a virtual enviroment [how to create a virtual enviroment](http://docs.python-guide.org/en/latest/dev/virtualenvs/).
```
 virtualenv ~/BLCF_saliency
 source ~/BLCF_saliency/bin/activate
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
* Lastly, modify the paths to store the datasets and saliency predictions in  *src/dataset/config.py* file located in the folder named as the dataset. (default location is 'data/').

### How to run it
```
Usage: python evaluation.py [OPTIONS]

Options:
  --dataset TEXT     Selected dataset for extraction  (availables 'instre' (default), 'oxford', 'paris')
  --path_data TEXT   path to store models             (default 'data/')
  --layer TEXT       layer from vgg16                 (default 'conv5_1')
  --max_dim INTEGER  Max dimension of images          (default '340')
  --weighting TEXT   Spatial weighting scheme         (availables None (default), 'gaussian', 'l2norm', 'SALGAN')
  --global_search    Global Search for queries        
  --query_expansion  Apply Average Query Expansion    
```
