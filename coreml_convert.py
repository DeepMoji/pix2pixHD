import torch
from options.convert_options import ConvertOptions
from models.models import create_model
import coremltools as ct


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


def convert_model(net, test_data):
    net.eval()
    traced_model = torch.jit.trace(net, test_data)

    ct_model = ct.convert(
        traced_model,
        # inputs=[ct.TensorType(name="input1", shape=data['B'].shape)]  # name "input_1" is used in 'quickstart'
        inputs=[ct.ImageType(name="image_input", shape=test_data.shape)]  # name "input_1" is used in 'quickstart'
    )
    return ct_model

if __name__ == '__main__':
    print('Convert models')
    print('Usage:')
    print('coreml_convert --model_in PATH_G_MODEL --model_out PATH_OUT_MODEL --no_instance '
          '--label_nc 0 --resize_or_crop resize_and_crop --loadSize 256 --fineSize 256')

    # Get input parameters
    opt = get_parameters()

    # Define input data of dimension 1x3xfine_sizexfine_size
    data = torch.rand(1, 3, opt.fineSize, opt.fineSize)
    data = data.cuda()
    # Create model
    model = create_model(opt, for_conversion=True)
    # Convert model

    model.netG.eval()
    model.netG(data)

    if 0:
        torch.onnx.export(model.netG, data, "/home/ubuntu/deployment/pix2pixhd/checkpoints/label2city/pix2pix.onnx",
                      input_names=['model_input'])
        coreml_model = ct.converters.onnx.convert(model="/home/ubuntu/deployment/pix2pixhd/checkpoints/label2city/pix2pix.onnx")

        from PIL import Image
        example_image = Image.open("/home/ubuntu/code/pix2pixHD/datasets/toonify/test_A/00000_01.png").resize((256, 256))
        # Make a prediction using Core ML
        out_dict = model.predict({"model_input": example_image})

    # This requires changing ReflectionPad2 https://github.com/apple/coremltools/issues/855
    coreml_model = convert_model(model.netG, data)
    coreml_model.save(opt.model_out)

    pass