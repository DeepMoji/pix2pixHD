import coremltools as ct
from coremltools.models.neural_network import quantization_utils
from coremltools.models.neural_network import optimization_utils
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
import torch.optim as optim
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

from options.train_options import TrainOptions
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader

from models.models import create_model

# from ../minimalml import quantize_coreml_network
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/code')
from minimalml.mkml_quantization import quantize_coreml_network
# from minimalml.quantization_utils_copy import quantize_weights1
from minimalml.mkml_model_utilities import get_layer_weight_stats
from minimalml.mkml_model_utilities import create_detailed_quant_values
from minimalml.mkml_model_utilities import prunableLayers


# batch_size_test = 1

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10


class Net(nn.Module):
    """
    Toy neural network
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x)
        # return F.log_softmax(x)

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


def check_toy_model(model_name, dataset_images):

    model_in = ct.models.MLModel(model_name)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(dataset_images, train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor()])), batch_size=1)
    correct = 0
    total = 0
    for data, target in test_loader:
        output = model_in.predict({'image_input': Image.fromarray(np.uint8(data[0, 0, :, :].numpy() * 255))})
        output = output['72']
        if target.numpy()[0] == np.argmax(output):
            correct = correct + 1
        total = total + 1
        if total % 100 == 0:
            print(total)
    print('Accuracy is ', correct / total)
    return correct / total

def check_models():
    model_in = ct.models.MLModel('/Users/michaelko/Code/ngrok/checkpoints/toy/model_quant.mlmodel')
    # model_in = ct.models.MLModel('/Users/michaelko/Code/ngrok/checkpoints/toy/model_full_norm_simple.mlmodel')

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/Users/michaelko/Code/ngrok/mnist/images', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor()])), batch_size=1)
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
        output = output['72']
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
    model_fp16 = quantization_utils.quantize_weights(model_in, nbits=3)
    model_fp16.save('/Users/michaelko/Code/ngrok/checkpoints/toy/model_quant3.mlmodel')

    return

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

def compute_gradients(names_list):
    """
    The function computes average gradient magnitude for all weights
    :param: names_list is a list of names as obtained from coreml stats get_layer_weight_stats(model_in) function
            The list may include many empty cells. The list has a member for each network sub part, even such as
            activation and dropout. Obviously, many of them are empty
            names_list
    [['bias', 'weights'], [], [], ['bias', 'weights'], [], [], [], ['bias', 'weights'], [], ['bias', 'weights'], [], []]
    :return:
    """

    # Get the number of non zero parts
    non_zero_num = [1 if len(name) > 0 else 0 for name in names_list]
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/Users/michaelko/Code/ngrok/mnist/images', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.5,), (0.5,))
                                   ])),
        batch_size=1, shuffle=True)

    network = Net()
    network.load_state_dict(torch.load('/Users/michaelko/Code/ngrok/checkpoints/toy/model_mnist.pth'))
    network.eval()

    # Get
    module_names = [k for k in network.named_children()]
    is_prunable = np.zeros(len(module_names))
    return_list = []
    for cnt, module in enumerate(module_names):
        found = False
        for k, good_module in enumerate(prunableLayers.classes):
            if isinstance(module[1], good_module):
                found = True
                break
        if found:
            is_prunable[cnt] = 1
            return_list.append(np.zeros(len(prunableLayers.translateToCoreML[k])))
        else:
            return_list.append([])
    # Check the number of prunable models
    if np.sum(is_prunable) != np.sum(np.array(non_zero_num)):
        print('PROBLEMS')

    # Go over all test data
    total_data = 0
    for data, target in test_loader:
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        total_data = total_data + 1
        if total_data % 150 == 0:
            print(total_data)
        # for each image, compute the average value of abs(gradient * w) for all weights and for all biases
        for k in range(len(is_prunable)):
            if is_prunable[k] == 0:
                continue
            # Get the type
            if type(module_names[k][1]) not in prunableLayers.classes:
                raise Exception('No such network class')
            ind = prunableLayers.classes.index(type(module_names[k][1]))
            # Go over all appropriate network parts
            for p_cnt, part_name in enumerate(prunableLayers.translateToCoreML[ind]):
                value = getattr(getattr(network, module_names[k][0]), part_name)
                return_list[k][p_cnt] = return_list[k][p_cnt] + np.mean(np.abs(value.detach() * value.grad.detach()).numpy())

    for k in range(len(return_list)):
        if len(return_list[k]) == 0:
            continue
        return_list[k] = return_list[k] / total_data

    return return_list

def compute_toon_gradients(names_list):
    # Get the number of non zero parts
    non_zero_num = [1 if len(name) > 0 else 0 for name in names_list]

    opt = TrainOptions().parse()
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    model = create_model(opt)

    module_names = [k for k in model.netG.named_children()]
    len(module_names[0][1]) # These are the names

    cnt = 0
    for param in model.netG.parameters():
        cnt = cnt + 1
        print(param)
    print('There are params ', cnt)

    cnt = 0
    for param in model.netG.named_parameters():
        cnt = cnt + 1
        print(param)

    print('There are params ', cnt)

    for cnt, module in enumerate(model.netG.modules()):
        print(cnt)

    model.eval()
    for i, data in enumerate(dataset):
        losses, generated = model(Variable(data['label']), Variable(data['inst']),
                                  Variable(data['image']), Variable(data['feat']), infer=False)

        # sum per device losses
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        loss_dict = dict(zip(model.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0)

    print(1)


def test(network, test_loader, test_losses):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def train(epoch, network, train_loader, train_losses, train_counter, optimizer):
    # specify you will be training
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
        train_counter.append((batch_idx*batch_size_train) + ((epoch-1)*len(train_loader.dataset)))
        torch.save(network.state_dict(), '/Users/michaelko/Code/ngrok/checkpoints/toy/model_mnist.pth')
        # torch.save(optimizer.state_dict(), '/home/ubuntu/code/mnist/results/optimizer.pth')

def train_loop():
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/Users/michaelko/Code/ngrok/mnist/images', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           # (0.1307,), (0.3081,))
                                           (0.5,), (0.5,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/Users/michaelko/Code/ngrok/mnist/images', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.5,), (0.5,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)

    network = Net()
    # network = SimpleNet()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    test(network, test_loader, test_losses)
    for epoch in range(1, n_epochs + 1):
        train(epoch, network, train_loader, train_losses, train_counter, optimizer)
        test(network, test_loader, test_losses)

def quantize_toy_model():
    model_in = ct.models.MLModel('/Users/michaelko/Code/ngrok/checkpoints/toy/model_quant.mlmodel')
    layers = get_layers(model_in)
    layer_types = get_layer_types(layers)
    stats, names_list = get_layer_weight_stats(model_in)
    quant_dict = create_detailed_quant_values(stats, names_list)
    qmodel = quantize_coreml_network(model_in, quant_dict)
    qmodel.save('/Users/michaelko/Code/ngrok/checkpoints/toy/model_quant1.mlmodel')

    check_toy_model('/Users/michaelko/Code/ngrok/checkpoints/toy/model_quant1.mlmodel',
                    '/Users/michaelko/Code/ngrok/mnist/images')

def prune_tutorial():
    network = Net()
    # network = SimpleNet()
    # network.load_state_dict(torch.load('/home/ubuntu/code/mnist/results/model.pth'))
    network.load_state_dict(torch.load('/Users/michaelko/Code/ngrok/checkpoints/toy/model_mnist.pth'))

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/Users/michaelko/Code/ngrok/mnist/images', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           mean=(0.5,), std=(0.5,))
                                   ])),
        batch_size=batch_size_test)
    test_losses = []
    test(network, test_loader, test_losses)

    test_data = torch.rand(1, 1, 28, 28)
    network.eval()
    traced_model = torch.jit.trace(network, test_data)

    ct_model = ct.convert(
        traced_model,
        # inputs=[ct.TensorType(name="input1", shape=data['B'].shape)]  # name "input_1" is used in 'quickstart'
        # inputs=[ct.ImageType(name="image_input", shape=test_data.shape)]  # name "input_1" is used in 'quickstart'
        # name "input_1" is used in 'quickstart'
        # inputs=[ct.ImageType(name="image_input", shape=test_data.shape, bias=0.1307/0.3081,
        #                      scale=1.0 / (255 * 0.3081), color_layout='G')]
        inputs=[ct.ImageType(name="image_input", shape=test_data.shape, bias=-1.0, scale=1.0/127.5, color_layout='G')]
    )
    # ct_model.save('/home/ubuntu/code/mnist/results/model_full.mlmodel')
    ct_model.save('/Users/michaelko/Code/ngrok/checkpoints/toy/model_quant.mlmodel')
    # ct_model.save('/home/ubuntu/code/mnist/results/model_full_norm_simple_noscale.mlmodel')

    pass

def quantize_toonify_models(method):

    model_in = ct.models.MLModel('/Users/michaelko/Code/ngrok/checkpoints/final_model/100_net_G_512.mlmodel')
    stats, names_list = get_layer_weight_stats(model_in)

    layer_types = get_layer_types(get_layers(model_in))
    non__zero_type = []
    non__zero_val = []
    for layer, stat in zip(layer_types, stats):
        if stat == []:
            continue
        non__zero_type.append(layer)
        non__zero_val.append(np.sum(stat[0, :]))

    if method == 'weight_based':
        print('Start weight based pruning')
        quant_dict = create_detailed_quant_values(stats, names_list, 0.1785, 3, 7)
        qmodel = quantize_coreml_network(model_in, quant_dict)
        qmodel.save('/Users/michaelko/Code/ngrok/checkpoints/label2city/modle_100_512_q14.mlmodel')

        # Test

    else:
        print('Start gradient based pruning')
        compute_toon_gradients(names_list)

    print('Done')

if __name__ == '__main__':
    print('Pruning test')
    # train_loop()

    # ---------------------------------------------------------------------------------------------------------------- #
    method = 'gradient_based'
    # method = 'weight_based'
    quantize_toonify_models(method)
    # ---------------------------------------------------------------------------------------------------------------- #

    # ---------------------------------------------------------------------------------------------------------------- #
    # model_in = ct.models.MLModel('/Users/michaelko/Code/ngrok/checkpoints/toy/model_quant.mlmodel')
    # layers = get_layers(model_in)
    # layer_types = get_layer_types(layers)
    # stats, names_list = get_layer_weight_stats(model_in)
    # # This approach is based on gradient weighted pruning. The simple pruning is in quantize_toy_model()
    # weighted_gradients = compute_gradients(names_list)
    # # The typical stat format is 2D array row0 - number of weight, row1 - sum of values
    # # Convert weighted gradients into 2D stats format
    # stat_weighted_grad = []
    # non_zero_stat_id = -1
    # non_zero_wg_id = -1
    # for k in range(len(stats)):
    #     if len(stats[k]) == 0:
    #         stat_weighted_grad.append([])
    #         continue
    #     non_zero_stat_id = non_zero_stat_id + 1
    #     for m in range(non_zero_wg_id + 1, len(weighted_gradients)):
    #         if len(weighted_gradients[m]) > 0:
    #             non_zero_wg_id = m
    #             break
    #     array2D = np.zeros((2, len(weighted_gradients[non_zero_wg_id])))
    #     array2D[0, :] = stats[k][0, :]
    #     array2D[1, :] = stats[k][0, :] * weighted_gradients[non_zero_wg_id]
    #     stat_weighted_grad.append(array2D)
    # quant_dict = create_detailed_quant_values(stat_weighted_grad, names_list)
    # qmodel = quantize_coreml_network(model_in, quant_dict)
    # qmodel.save('/Users/michaelko/Code/ngrok/checkpoints/toy/model_quant1.mlmodel')
    #
    # check_toy_model('/Users/michaelko/Code/ngrok/checkpoints/toy/model_quant1.mlmodel',
    #                 '/Users/michaelko/Code/ngrok/mnist/images')
    # ---------------------------------------------------------------------------------------------------------------- #

    # create_models()
    # quantize_toy_model()
    # check_models()
    # prune_tutorial()
    # quantize_toonify_models()

    # '/Users/michaelko/Code/ngrok/checkpoints/toy/model_full.mlmodel'
    # /Users/michaelko/Code/ngrok/checkpoints/toy/model_quant8.mlmodel

    # check_toy_model('/Users/michaelko/Code/ngrok/checkpoints/toy/model_quant3.mlmodel',
    #                 '/Users/michaelko/Code/ngrok/mnist/images')

