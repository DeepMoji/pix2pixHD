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
from face_postprocessing import sharpen_face
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


def build_web_page():
    webpage = html.HTML('/Users/michaelko/Code/ngrok/results', 'results')

    visuals = OrderedDict([('FFHQ_Original image', cv.imread('/Users/michaelko/Downloads/mk_results_aligned_ffhq_00537_01.png')),
                           ('FFHQ_SLOW TOON', cv.imread('/Users/michaelko/Downloads/mk_results_tooned_images_00537_01-toon.jpg')),
                           ('FFHQ_DeepMoji 1024_1', cv.imread('/Users/michaelko/Downloads/out_90_1024.png')),
                           ('FFHQ_DeepMoji 1024_2', cv.imread('/Users/michaelko/Downloads/mkres_20_1024.png')),
                           ('FFHQ_DeepMoji 512_1', cv.imread('/Users/michaelko/Downloads/mkres_160_512.png')),
                           ('FFHQ_DeepMoji 512_2', cv.imread('/Users/michaelko/Downloads/mkres_60_512.png')),
                           ('FFHQ_API', cv.imread('/Users/michaelko/Downloads/output_00537.jpg'))])
    save_images(webpage, visuals)

    visuals = OrderedDict(
        [('Original image', cv.imread('/Users/michaelko/Downloads/test_photo.png')),
         ('SLOW TOON', cv.imread('/Users/michaelko/Downloads/slow_toon.jpg')),
         ('DeepMoji 1024_1', cv.imread('/Users/michaelko/Downloads/test_photo_90_1024.png')),
         ('DeepMoji 1024_2', cv.imread('/Users/michaelko/Downloads/test_photo_20_1024.png')),
         ('DeepMoji 512_1', cv.imread('/Users/michaelko/Downloads/test_photo_160_512.png')),
         ('DeepMoji 512_2', cv.imread('/Users/michaelko/Downloads/test_photo_60_512.png')),
         ('API', cv.imread('/Users/michaelko/Downloads/test_photo_api_out.jpg'))])
    save_images(webpage, visuals)

    webpage.save()

def compare_models_1():
    pass


def compare_models():
    img_folder = '/Users/michaelko/Code/ngrok/images_test'
    img_api_folder = '/Users/michaelko/Code/ngrok/images_test_res'
    model_folder = '/Users/michaelko/Code/ngrok/checkpoints/label2city'
    res_dir = '/Users/michaelko/Code/ngrok/res'

    web_dir = os.path.join(res_dir, 'comparison')
    webpage = html.HTML(web_dir, 'Comparison')

    # visuals = OrderedDict([('input_label', img1), ('synthesized_image', img2)])
    model_files = []    #[f for f in listdir(model_folder) if isfile(join(model_folder, f))]

    for f in listdir(model_folder):
        if not isfile(join(model_folder, f)):
            continue
        if f[0] == '.':
            continue
        model_files.append(f)

    # Open models
    models = []
    for model_name in model_files:
        if model_name[0] == '.':
            continue
        print('Loading ', model_name)
        models.append(ct.models.MLModel(join(model_folder, model_name)))

    img_files = [f for f in listdir(img_folder) if isfile(join(img_folder, f))]
    cnt = 0
    for img_name in img_files:
        if img_name[0] == '.':
            continue
        print('Start image ', img_name, ' ', cnt)
        cnt = cnt + 1
        img_orig = Image.open(join(img_folder, img_name))
        results = []
        for model in models:
            desc = model.get_spec().description
            img = img_orig.resize((desc.input[0].type.imageType.height, desc.input[0].type.imageType.width))
            res = model.predict({'image_input': img})
            r = res[list(res.keys())[0]][0, 0, :, :]
            g = res[list(res.keys())[0]][0, 1, :, :]
            b = res[list(res.keys())[0]][0, 2, :, :]
            rgb = np.zeros((desc.input[0].type.imageType.height, desc.input[0].type.imageType.width, 3))
            rgb[:, :, 0] = 0.5*(b + 1.0)
            rgb[:, :, 1] = 0.5*(g + 1.0)
            rgb[:, :, 2] = 0.5*(r + 1.0)
            results.append(rgb*255)

        api_img_name = img_name[0:-7] + '.jpg'
        visuals = OrderedDict([('orig' + img_name,cv.cvtColor(np.array(img_orig), cv.COLOR_RGB2BGR)),
                               ('api' + img_name, cv.imread(join(img_api_folder, api_img_name))),
                                (model_files[0][:-7] + img_name, results[0]),
                                (model_files[1][:-7] + img_name, results[1]),
                                (model_files[2][:-7] + img_name, results[2]),
                                (model_files[3][:-7] + img_name, results[3]),
                                (model_files[4][:-7] + img_name, results[4])])

        # visuals = OrderedDict([(model_files[0][:-7] + img_name, results[0]),
        #                        (model_files[1][:-7] + img_name, results[1]),
        #                        (model_files[2][:-7] + img_name, results[2]),
        #                        (model_files[3][:-7] + img_name, results[3]),
        #                        (model_files[4][:-7] + img_name, results[4]),
        #                        ('orig' + img_name,cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR))])
        # visuals = OrderedDict([(model_files[0][:-7] + img_name, results[0]),
        #                        (model_files[1][:-7] + img_name, results[1]),
        #                        ('orig' + img_name, cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR))])
        save_images(webpage, visuals)

    webpage.save()


def compare_models1():
    img_folder = '/Users/michaelko/Code/ngrok/images_test'
    img_api_folder = '/Users/michaelko/Code/ngrok/images_test_res'
    model_folder = '/Users/michaelko/Code/ngrok/checkpoints/final_model'
    res_dir = '/Users/michaelko/Code/ngrok/res1'

    web_dir = os.path.join(res_dir, 'comparison')
    webpage = html.HTML(web_dir, 'Comparison')

    # visuals = OrderedDict([('input_label', img1), ('synthesized_image', img2)])
    model_files = []    #[f for f in listdir(model_folder) if isfile(join(model_folder, f))]

    for f in listdir(model_folder):
        if not isfile(join(model_folder, f)):
            continue
        if f[0] == '.':
            continue
        model_files.append(f)

    # Open models
    models = []
    for model_name in model_files:
        if model_name[0] == '.':
            continue
        print('Loading ', model_name)
        models.append(ct.models.MLModel(join(model_folder, model_name)))

    img_files = [f for f in listdir(img_folder) if isfile(join(img_folder, f))]
    cnt = 0
    for img_name in img_files:
        if img_name[0] == '.':
            continue
        print('Start image ', img_name, ' ', cnt)
        cnt = cnt + 1
        img_orig = Image.open(join(img_folder, img_name))
        results = []
        for model in models:
            desc = model.get_spec().description
            img = img_orig.resize((desc.input[0].type.imageType.height, desc.input[0].type.imageType.width))
            res = model.predict({'image_input': img})
            r = res[list(res.keys())[0]][0, 0, :, :]
            g = res[list(res.keys())[0]][0, 1, :, :]
            b = res[list(res.keys())[0]][0, 2, :, :]
            rgb = np.zeros((desc.input[0].type.imageType.height, desc.input[0].type.imageType.width, 3))
            rgb[:, :, 0] = 0.5*(b + 1.0)
            rgb[:, :, 1] = 0.5*(g + 1.0)
            rgb[:, :, 2] = 0.5*(r + 1.0)
            results.append(rgb*255)
            results.append(sharpen_face(results[0]))

        api_img_name = img_name[0:-7] + '.jpg'
        visuals = OrderedDict([('orig' + img_name,cv.cvtColor(np.array(img_orig), cv.COLOR_RGB2BGR)),
                               ('api' + img_name, cv.imread(join(img_api_folder, api_img_name))),
                                (model_files[0][:-7] + img_name, results[0]),
                                ('sharp.' + model_files[0][:-7] + img_name, results[1])])

        # visuals = OrderedDict([(model_files[0][:-7] + img_name, results[0]),
        #                        (model_files[1][:-7] + img_name, results[1]),
        #                        (model_files[2][:-7] + img_name, results[2]),
        #                        (model_files[3][:-7] + img_name, results[3]),
        #                        (model_files[4][:-7] + img_name, results[4]),
        #                        ('orig' + img_name,cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR))])
        # visuals = OrderedDict([(model_files[0][:-7] + img_name, results[0]),
        #                        (model_files[1][:-7] + img_name, results[1]),
        #                        ('orig' + img_name, cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR))])
        save_images(webpage, visuals)

    webpage.save()

def single_test():
    pass
    img_name = '/Users/michaelko/Downloads/test_photo.png'
    model_name = '/Users/michaelko/Code/ngrok/checkpoints/label2city/20_net_G_1024.mlmodel'
    model = ct.models.MLModel(model_name)

    desc = model.get_spec().description
    # spec.WhichOneof()

    img = Image.open(img_name)
    img = img.resize((desc.input[0].type.imageType.height, desc.input[0].type.imageType.width))

    res = model.predict({'image_input': img})
    r = res[list(res.keys())[0]][0, 0, :, :]
    g = res[list(res.keys())[0]][0, 1, :, :]
    b = res[list(res.keys())[0]][0, 2, :, :]
    rgb = np.zeros((desc.input[0].type.imageType.height, desc.input[0].type.imageType.width, 3))
    rgb[:, :, 0] = 0.5 * (b + 1.0)
    rgb[:, :, 1] = 0.5 * (g + 1.0)
    rgb[:, :, 2] = 0.5 * (r + 1.0)

    cv.imwrite('/Users/michaelko/Downloads/test_photo_20_1024.png', 255*rgb)


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
    # convert_32_to_16_1()
    compare_models1()
    # single_test()
    # build_web_page()
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