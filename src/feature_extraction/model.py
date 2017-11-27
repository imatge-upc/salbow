import os, glob, sys
from keras.applications import vgg16
from utils import *


##########################################################################
# Extract images for X dataset.
#
# Sets the larges dimension to 340. :)
###########################################################################

def init_model( layer='conv5_1' ):
    #init model
    model = vgg16.VGG16(include_top=False, weights='imagenet')

    if layer == 'pool1':
        # remove layers
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()

        model.layers.pop()
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()

        model.layers.pop()
        model.layers.pop()
        model.layers.pop()

        model.layers.pop()
        model.layers.pop()
        model.layers.pop()

        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []


    if layer == 'pool2':
        # remove layers
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()

        model.layers.pop()
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()

        model.layers.pop()
        model.layers.pop()
        model.layers.pop()


        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []

    if layer == 'pool3':
        # remove layers
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()

        model.layers.pop()
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()

        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []


    if layer == 'pool4':
        # remove layers
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()

        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []

    if layer == 'conv5_1':
        # remove layers
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []
    elif layer == 'conv5_2':
        # remove layers
        model.layers.pop()
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []
    elif layer == 'conv5_3':
        # remove layers
        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []

    return model
