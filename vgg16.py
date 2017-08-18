import os
import sys

from parrots.dnn import GModule
from parrots.dnn.layerprotos import (Accuracy, Convolution, Deconvolution, Dropout,
                                       FullyConnected, ReLU, Pooling, Softmax,
                                       SoftmaxWithLoss)
from parrots import base


def conv_relu(x, kernel_size, num_outputs, stride=1, pad=0, dilation=1,
              w_policy=None, b_policy=None, name=None):  # yapf: disable
    x = x.to(Convolution(
        kernel_size, num_outputs, stride=stride, pad=pad, hole=dilation,
        w_policy=w_policy, b_policy=b_policy),
             name='conv' + name)
    x = x.to(ReLU(), inplace=True, name='relu' + name)
    return x


def create_model():
    mod = GModule('vgg16')
    inputs = {
        'data': 'float32(480, 480, 3, _)',
        'label': 'uint32(480, 480, 1, _)',
        'label_weight': 'float32(480, 480, 1, _)',
    }
    mod.input_slots = tuple(inputs.keys())
    x = mod.var('data')

    # conv 1
    x = conv_relu(x, 3, 64, pad=1, name='1_1')
    x = conv_relu(x, 3, 64, pad=1, name='1_2')
    x = x.to(Pooling('max', 2, stride=2), name='pool1')

    # conv2
    x = conv_relu(x, 3, 128, pad=1, name='2_1')
    x = conv_relu(x, 3, 128, pad=1, name='2_2')
    x = x.to(Pooling('max', 2, stride=2), name='pool2')

    # conv3
    x = conv_relu(x, 3, 256, pad=1, name='3_1')
    x = conv_relu(x, 3, 256, pad=1, name='3_2')
    x = conv_relu(x, 3, 256, pad=1, name='3_3')
    x = x.to(Pooling('max', 2, stride=2), name='pool3')

    # conv4 
    # set pad to be 1, feature map does not shrink
    x = conv_relu(x, 3, 512, pad=1, name='4_1')
    x = conv_relu(x, 3, 512, pad=1, name='4_2')
    x = conv_relu(x, 3, 512, pad=1, name='4_3')
    x = x.to(Pooling('max', 3, pad=1, stride=1), name='pool4')

    # conv5, use atrous convolution to dense extract features
    x = conv_relu(x, 3, 512, pad=2, dilation=2, name='5_1')
    x = conv_relu(x, 3, 512, pad=2, dilation=2, name='5_2')
    x = conv_relu(x, 3, 512, pad=2, dilation=2, name='5_3')
    x = x.to(Pooling('max', 3, pad=1, stride=1), name='pool5')
    x = x.to(Pooling('ave', 3, pad=1, stride=1), name='pool5a')

    # conv_fc6, atrous convolution
    x = conv_relu(x, 7, 4096, pad=12, stride=1, dilation=4, name='_fc6')
    x = x.to(Dropout(0.5), inplace=True, name='drop6')

    # conv_fc7
    x = conv_relu(x, 1, 4096, name='_fc7')
    x = x.to(Dropout(0.5), inplace=True, name='drop7')

    x = x.to(Convolution(1, 21, w_policy={'init': 'gauss(0.01)'}, b_policy={'init': 'fill(0)'}),  name='conv_fc8')

    # deconvolution (bilinear interpolation) used to upsampling
    x = x.to(Deconvolution(16, 21, pad=4, stride=8, bias=False,
                           w_policy={'init': 'fill(1)', 'lr_mult': '0', 'decay_mult': '0'}),
              name='upscore2')

    # softmax layer
    mod.vars(x, 'label', 'label_weight').to(SoftmaxWithLoss(axis=2), name='loss')
    model = mod.compile(inputs=inputs, seal=False)
    model.add_flow('main',
                   inputs.keys(), ['loss', 'accuracy_top1', 'accuracy_top5'],
                   ['loss'])
    model.seal()
    return model


if __name__ == '__main__':
    base.set_debug_log(True)
    model = create_model()
    with open('model.yaml', 'w+') as f:
        f.write(model.to_yaml_text())
