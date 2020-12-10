import coremltools as ct
from coremltools.models.neural_network import quantization_utils
import coremltools.models.neural_network
import os
from util import html
import util.util as util
from collections import OrderedDict
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np
import cv2 as cv



def toy_problem():
    pass


def get_layer_types(layers):
    layer_types = []
    for layer in layers:
        layer_type = layer.WhichOneof("layer")
        layer_types.append(layer_type)
    return layer_types

def adaptve_quantization_example():
    pass

def get_layers(model):
    layers = model.get_spec().neuralNetwork.layers
    return layers

if __name__ == '__main__':
    print('Pruning test')

    model_in = ct.models.MLModel('/Users/michaelko/Code/ngrok/checkpoints/toy/model_full.mlmodel')
    # model_in = ct.models.MLModel('/Users/michaelko/Code/ngrok/checkpoints/label2city.old/70_net_G_16.mlmodel')
    layers = get_layers(model_in)
    layer_types = get_layer_types(layers)
    print(layer_types)
    model_fp16 = quantization_utils.quantize_weights(model_in, nbits=6)

    # Example 1 - quantize weights differently for each layers
    num_layers = len(layer_types)
    nbits = np.random.randint(8, size=num_layers) + 1
    # 1. quantize_weights (model)
    #    spec = full_precision_model.get_spec()
    # 2. _quantize_spec_weights(spec, nbits, qmode, **kwargs)
    # 3. _quantize_nn_spec(spec.neuralNetwork, nbits, quantization_mode, **kwargs)
    #      nn_spec = spec.neuralNetwork
    #      layers = nn_spec.layers
    #       for layer in layers
    #           _quantize_wp_field(
    #                     layer.convolution.weights, nbits, qm, shape, **kwargs
    #                 )

