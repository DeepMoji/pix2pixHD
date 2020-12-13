import coremltools as ct
from coremltools.models.neural_network import quantization_utils
import coremltools.models.neural_network
from util import html
import util.util as util
from collections import OrderedDict
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np
import cv2 as cv
import os, sys

import torch
import torchvision
import torch.nn.functional as F

# from ../minimalml import quantize_coreml_network
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minimalml.mkml_quantization import quantize_coreml_network


batch_size_test = 1


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


def check_models():
    # model_in = ct.models.MLModel('/Users/michaelko/Code/ngrok/checkpoints/toy/model_quant8.mlmodel')
    model_in = ct.models.MLModel('/Users/michaelko/Code/ngrok/checkpoints/toy/model_full_norm_simple.mlmodel')

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/Users/michaelko/Code/ngrok/mnist/images', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor()])), batch_size=batch_size_test)
        # torchvision.datasets.MNIST('/Users/michaelko/Code/ngrok/mnist/images', train=False, download=True,
        #                        transform=torchvision.transforms.Compose([
        #                            torchvision.transforms.ToTensor(),
        #                            torchvision.transforms.Normalize(
        #                                (0.1307,), (0.3081,))
        #                        ])), batch_size = batch_size_test, shuffle = True)
    test_losses = []
    test_loss = 0
    correct = 0
    total = 0
    # pred = np.load('/Users/michaelko/Code/ngrok/checkpoints/toy/pred1000.npy')
    for data, target in test_loader:
        output = model_in.predict({'image_input': Image.fromarray(np.uint8(data[0, 0, :, :].numpy() * 255))})
        output = output['70']
        if target.numpy()[0] == np.argmax(output):
            correct = correct + 1
        # if np.argmax(output) != pred[total]:
        #     print(pred[total])
        total = total + 1
        if total % 100 == 0:
            print(total)
        # test_loss += F.nll_loss(output, target, size_average=False).item()
        # pred = output.data.max(1, keepdim=True)[1]
        # correct += pred.eq(target.data.view_as(pred)).sum()
    # test_loss /= len(test_loader.dataset)
    # test_losses.append(test_loss)
    # print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    print('Accuracy is ', correct / total)

def create_models():
    print('Pruning test')

    model_in = ct.models.MLModel('/Users/michaelko/Code/ngrok/checkpoints/toy/model_full.mlmodel')
    # model_in = ct.models.MLModel('/Users/michaelko/Code/ngrok/checkpoints/label2city.old/70_net_G_16.mlmodel')
    layers = get_layers(model_in)
    layer_types = get_layer_types(layers)
    print(layer_types)
    model_fp16 = quantization_utils.quantize_weights(model_in, nbits=8)
    model_fp16.save('/Users/michaelko/Code/ngrok/checkpoints/toy/model_quant8.mlmodel')

    # Example 1 - quantize weights differently for each layers
    num_layers = len(layer_types)
    nbits = np.random.randint(1, size=num_layers) + 7
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
    model_in = quantize_coreml_network(model_in, nbits)
    model_in.save('/Users/michaelko/Code/ngrok/checkpoints/toy/model_quant.mlmodel')

    # Test
    model_in = ct.models.MLModel('/Users/michaelko/Code/ngrok/checkpoints/toy/model_quant8.mlmodel')
    pass


if __name__ == '__main__':
    print('Pruning test')
    #    create_models()
    check_models()

