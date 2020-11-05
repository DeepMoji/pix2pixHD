import argparse
import torch
from options.convert_options import ConvertOptions
from models.models import create_model


def get_parameters():
    """
    The function returns the parameters used for creating the coreml model
    :return:
    """
    # Get model options
    opt = ConvertOptions().parse(save=False)
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    return opt


if __name__ == '__main__':
    print('Convert models')
    print('Usage:')
    print('coreml_convert --model_in PATH_G_MODEL --model_out PATH_OUT_MODEL --no_instance '
          '--label_nc 0 --resize_or_crop resize_and_crop --loadSize 256 --fineSize 256')

    # Get input parameters
    opt = get_parameters()

    # Define input data of dimension 1x3xfine_sizexfine_size
    data = torch.rand(1, 3, opt.fineSize, opt.fineSize)
    # Create model
    model = create_model(opt, for_conversion=True)



    pass