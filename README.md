# Saliency Weighted Convolutional features for Instance Search


| ![Eva Mohedano][EvaMohedano-photo] |  ![Kevin McGuinness][KevinMcGuinness-photo] |  ![Xavier Giro-i-Nieto][XavierGiro-photo] | ![Noel O'Connor][NoelOConnor-photo]  | 
|:-:|:-:|:-:|:-:|
| [Eva Mohedano](https://www.insight-centre.org/users/eva-mohedano)  |  [Kevin McGuinness](https://www.insight-centre.org/users/eva-mohedano)   | [Xavier Giro-i-Nieto](https://imatge.upc.edu/web/people/xavier-giro)   | [Noel O'Connor](https://www.insight-centre.org/users/noel-oconnor)   | 


[EvaMohedano-photo]: https://raw.githubusercontent.com/imatge-upc/retrieval-2016-lostobject/master/authors/Eva.jpg?token=AKsMd4iuttxHH44mYL3mPpJEtSvXVXF8ks5Xe-AWwA%3D%3D "Eva Mohedano" 
[KevinMcGuinness-photo]: https://raw.githubusercontent.com/imatge-upc/retrieval-2016-lostobject/master/authors/Kevin.jpg?token=AKsMd4VU31T7Bh8CztufWEWNudazbB_Uks5Xe-AxwA%3D%3D "Kevin McGuinness"
[XavierGiro-photo]: https://raw.githubusercontent.com/evamohe/BoW_CNN_InstanceSearch/master/authors/giro.jpg?token=AHPpwDdVdPYfMIwMBgHbjK9pPMJva1GOks5X1vHIwA%3D%3D "Xavier Giro-i-Nieto"
[NoelOConnor-photo]: https://raw.githubusercontent.com/imatge-upc/retrieval-2016-lostobject/master/authors/Noel.jpg?token=AKsMdyemO5eJke9B9rqdRtA7otJscq1wks5Xe-BEwA%3D%3D "Noel O'Connor"

A joint collaboration between:

| ![logo-insight] | ![logo-dcu] | ![logo-upc] | ![logo-etsetb] | ![logo-gpi] | 
|:-:|:-:|:-:|:-:|:-:|
| [Insight Centre for Data Analytics](insight-web) | [Dublin City University (DCU)](dcu-web)  |[Universitat Politecnica de Catalunya (UPC)](upc-web)   | [UPC ETSETB TelecomBCN](etsetb-web)  | [UPC Image Processing Group](gpi-web) | 

[insight-web]: https://www.insight-centre.org/ 
[dcu-web]: http://www.dcu.ie/
[upc-web]: http://www.upc.edu/?set_language=en 
[etsetb-web]: https://www.etsetb.upc.edu/en/ 
[gpi-web]: https://imatge.upc.edu/web/ 


[logo-insight]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/insight.jpg "Insight Centre for Data Analytics"
[logo-dcu]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/dcu.png "Dublin City University"
[logo-upc]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/upc.jpg "Universitat Politecnica de Catalunya"
[logo-etsetb]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/etsetb.png "ETSETB TelecomBCN"
[logo-gpi]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/gpi.png "UPC Image Processing Group"

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
Dataset images are automatically downloaded in:
```
PATH_DATASET='custom/dataset/path'
```
[Precomputed data](https://drive.google.com/drive/folders/18NmIcyEIJ8p9GO14rUB3n3wTnx8pezt_) contained saliency predictions for the three datasets, and BLCF models, assignment maps and raw features.

saliency masks for each of the datasets are stored in:
```
PATH_SALIENCY='custom/saliency/path'
```
BLCF models, features and assignment maps are stored in:
```
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
to the assignment maps of Instre, reporting mAP when performing query expansion (top 10 retrieved images). Results using [precomputed data](https://drive.google.com/drive/folders/18NmIcyEIJ8p9GO14rUB3n3wTnx8pezt_).


## Acknowledgements

|   |   |
|:--|:-:|
|  This publication has emanated from research conducted with the financial support of Science Foundation Ireland (SFI) under grant number SFI/12/RC/2289 and SFI/15/SIRG/3283. |  ![logo-ireland] |
|  The Image ProcessingGroup at the UPC is a [SGR14 Consolidated Research Group](https://imatge.upc.edu/web/projects/sgr14-image-and-video-processing-group) recognized and sponsored by the Catalan Government (Generalitat de Catalunya) through its [AGAUR](http://agaur.gencat.cat/en/inici/index.html) office. |  ![logo-catalonia] |
|  This work has been developed in the framework of the project [BigGraph TEC2013-43935-R](https://imatge.upc.edu/web/projects/biggraph-heterogeneous-information-and-graph-signal-processing-big-data-era-application), funded by the Spanish Ministerio de Economía y Competitividad and the European Regional Development Fund (ERDF).  | ![logo-spain] | 

[logo-ireland]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/sfi.png "Logo of Science Foundation Ireland"
[logo-catalonia]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/generalitat.jpg "Logo of Catalan government"
[logo-spain]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/MEyC.png "Logo of Spanish government"


## Contact

If you have any general doubt about our work or code which may be of interest for other researchers, please use the [public issues section](https://github.com/imatge-upc/salbow/issues) on this github repo. Alternatively, drop us an e-mail at <mailto:eva.mohedano@insight-centre.org>.
