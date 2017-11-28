# dataset images are automatically downloaded in:
PATH_DATASET='datasets/'


# saliency masks for each of the datasets are stored in:
PATH_SALIENCY='saliency/'
'''
Download pre-computed saliency masks from:
'https://drive.google.com/drive/folders/18NmIcyEIJ8p9GO14rUB3n3wTnx8pezt_?usp=sharing'"
Folder structure:
    PATH_SALIENCY/[dataset]/[mask]
'''

# Available saliency maks
MASKS=['itti', 'bms', 'salnet', 'SALGAN', 'SAM_VGG16', 'SAM_VGG_ResNet']

# BLCF models, features and assignment maps are stored in:
PATH_OUTPUT='output/'
'''
Download pre-computed assignment maps, raw features, and visual vocabularies from
'https://drive.google.com/drive/folders/18NmIcyEIJ8p9GO14rUB3n3wTnx8pezt_?usp=sharing'"
Folder structure:
    PATH_OUTPUT/[dataset]/assignments   <-- keyframes and query assignment maps
    PATH_OUTPUT/[dataset]/models        <-- PCA and Centroids
    PATH_OUTPUT/[dataset]/conv_features <-- Original Conv features from VGG16
'''
