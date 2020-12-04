import coremltools as ct
from coremltools.models.neural_network import quantization_utils
import os
from util import html
import util.util as util
from collections import OrderedDict
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np
import cv2 as cv
# from torch import nn
# import torch.nn.functional as F

# import torch


def convert_32_to_16_1():
    # model_in = ct.models.MLModel('/Users/michaelko/Code/pix2pixHD/checkpoints/label2city/160_net_G_512.mlmodel')
    model_in = ct.models.MLModel('/Users/michaelko/Code/ngrok/checkpoints/label2city/70_net_G.mlmodel')
    model_fp16 = quantization_utils.quantize_weights(model_in, nbits=8, quantization_mode="kmeans")
    model_fp16.save('/Users/michaelko/Code/ngrok/checkpoints/label2city/70_net_G_8_1.mlmodel')


def save_images(webpage, visuals):
    image_dir = webpage.get_image_dir()
    name = 'name'

    webpage.add_header(name)
    ims = []
    txts = []
    links = []

    for label, image_numpy in visuals.items():
        image_name = '%s_%s.jpg' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        cv.imwrite(save_path, image_numpy)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=256)

def compare_models():
    img_folder = '/Users/michaelko/Code/pix2pixHD/datasets/toonify/test'
    model_folder = '/Users/michaelko/Code/pix2pixHD/checkpoints/label2city'
    res_dir = '/Users/michaelko/Code/pix2pixHD/results'

    web_dir = os.path.join(res_dir, 'comparison')
    webpage = html.HTML(web_dir, 'Comparison')

    # visuals = OrderedDict([('input_label', img1), ('synthesized_image', img2)])
    model_files = [f for f in listdir(model_folder) if isfile(join(model_folder, f))]
    # Open models
    models = []
    for model_name in model_files:
        models.append(ct.models.MLModel(join(model_folder, model_name)))

    img_files = [f for f in listdir(img_folder) if isfile(join(img_folder, f))]
    for img_name in img_files:
        img = Image.open(join(img_folder, img_name))
        img = img.resize((512, 512))
        results = []
        for model in models:
            res = model.predict({'image_input': img})
            r = res['730'][0, 0, :, :]
            g = res['730'][0, 1, :, :]
            b = res['730'][0, 2, :, :]
            rgb = np.zeros((512, 512, 3))
            rgb[:, :, 0] = b
            rgb[:, :, 1] = g
            rgb[:, :, 2] = r
            results.append(rgb*255)

        visuals = OrderedDict([(model_files[0][:-7] + img_name, results[0]),
                               (model_files[1][:-7] + img_name, results[1]),
                               (model_files[2][:-7] + img_name, results[2]),
                               (model_files[3][:-7] + img_name, results[3]),
                               (model_files[4][:-7] + img_name, results[4]),
                               ('orig' + img_name,cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR))])
        save_images(webpage, visuals)

    webpage.save()


def single_test():
    pass
    img_name = '/Users/michaelko/Code/pix2pixHD/results/linux_160/images/12400_01_input_label.jpg'
    model_name = '/Users/michaelko/Code/pix2pixHD/checkpoints/label2city/latest_net_G.mlmodel'
    model = ct.models.MLModel(model_name)

    img = Image.open(img_name)
    img = img.resize((512, 512))

    res = model.predict({'image_input': img})
    r = res['730'][0, 0, :, :]
    g = res['730'][0, 1, :, :]
    b = res['730'][0, 2, :, :]
    rgb = np.zeros((512, 512, 3))
    rgb[:, :, 0] = 0.5 * (b + 1.0)
    rgb[:, :, 1] = 0.5 * (g + 1.0)
    rgb[:, :, 2] = 0.5 * (r + 1.0)

    cv.imwrite('/Users/michaelko/Downloads/example_011.png', 255*rgb)


# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         # 1 input image channel, 6 output channels, 3x3 square conv kernel
#         self.conv1 = nn.Conv2d(1, 6, 3)
#         self.conv2 = nn.Conv2d(6, 16, 3)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#         torch.nn.init.xavier_uniform_(self.conv1.weight)
#         torch.nn.init.xavier_uniform_(self.conv2.weight)
#         torch.nn.init.xavier_uniform_(self.fc1.weight)
#         torch.nn.init.xavier_uniform_(self.fc2.weight)
#         torch.nn.init.xavier_uniform_(self.fc3.weight)
#
#     def forward(self, x):
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = x.view(-1, int(x.nelement() / x.shape[0]))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
# def pruning_example():
#     pass
#     device = torch.device("cpu")
#
#     model = LeNet().to(device=device)
#     module = model.conv1
#     print(list(module.named_parameters()))
#
#     print(model)

if __name__ == '__main__':
    print('Convert models')
    print('Usage for pytorch to mlmodel:')
    print('coreml_convert --model_in PATH_G_MODEL --model_out PATH_OUT_MODEL --no_instance '
          '--label_nc 0 --resize_or_crop resize_and_crop --loadSize 512 --fineSize 512')

    print('Usage for f32 to f16 conversion:')
    print('coreml_convert --model_in PATH_G_MODEL --model_out PATH_OUT_MODEL --Convert32to16')

    # Get input parameters
    convert_32_to_16_1()
    # compare_models()
    # single_test()

    # pruning_example()


    # in_folder = "/Users/michaelko/Code/pix2pixHD/datasets/toonify/test"
    # out_folder = "/Users/michaelko/Code/pix2pixHD/datasets/toonify/test_jpg"
    #
    # onlyfiles = [f for f in listdir(in_folder) if isfile(join(in_folder, f))]
    #
    # for m_f in onlyfiles:
    #     img = cv.imread(join(in_folder, m_f))
    #     out_name = join(out_folder, m_f)
    #     cv.imwrite(out_name[:-3] + "jpg", img);
    #
    #
    #
    # pass