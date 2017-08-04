import os
import sys

parrots_home = os.environ.get('PARROTS_HOME')
if not parrots_home:
    raise EnvironmentError(
        'The environment variable "PARROTS_HOME" is not set.')
sys.path.append(os.path.join(parrots_home, 'parrots/python'))

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

    # convolution default: bias=True, pad=0, stride=1, hole=1, w_policy=None, b_policy=None)
    # pooling default: pad=0, stride=1
    x = conv_relu(x, 3, 64, pad=1, name='1_1')
    x = conv_relu(x, 3, 64, pad=1, name='1_2')
    x = x.to(Pooling('max', 2, stride=2), name='pool1')
    x = conv_relu(x, 3, 128, pad=1, name='2_1')
    x = conv_relu(x, 3, 128, pad=1, name='2_2')
    x = x.to(Pooling('max', 2, stride=2), name='pool2')
    x = conv_relu(x, 3, 256, pad=1, name='3_1')
    x = conv_relu(x, 3, 256, pad=1, name='3_2')
    x = conv_relu(x, 3, 256, pad=1, name='3_3')
    x = x.to(Pooling('max', 2, stride=2), name='pool3')
    x = conv_relu(x, 3, 512, pad=1, name='4_1')
    x = conv_relu(x, 3, 512, pad=1, name='4_2')
    x = conv_relu(x, 3, 512, pad=1, name='4_3')
    x = x.to(Pooling('max', 3, pad=1, stride=1), name='pool4')
    x = conv_relu(x, 3, 512, pad=2, dilation=2, name='5_1')
    x = conv_relu(x, 3, 512, pad=2, dilation=2, name='5_2')
    x = conv_relu(x, 3, 512, pad=2, dilation=2, name='5_3')
    x = x.to(Pooling('max', 3, pad=1, stride=1), name='pool5')
    x = x.to(Pooling('ave', 3, pad=1, stride=1), name='pool5a')

    # x = x.to(FullyConnected(4096), name='fc6')
    # x = x.to(ReLU(), inplace=True, name='relu6')
    x = conv_relu(x, 7, 4096, pad=12, stride=1, dilation=4, name='_fc6')
    x = x.to(Dropout(0.5), inplace=True, name='drop6')

    # x = x.to(FullyConnected(4096), name='fc7')
    # x = x.to(ReLU(), inplace=True, name='relu7')
    x = conv_relu(x, 1, 4096, name='_fc7')
    x = x.to(Dropout(0.5), inplace=True, name='drop7')


    # x = x.to(FullyConnected(1000), name='fc8')
    x = x.to(Convolution(1, 21, w_policy={'init': 'gauss(0.01)'}, b_policy={'init': 'fill(0)'}),  name='conv_fc8')

    # deconvolution
    x = x.to(Deconvolution(16, 21, pad=4, stride=8, bias=False,
                           w_policy={'init': 'fill(1)', 'lr_mult': '0', 'decay_mult': '0'}),
              name='upscore2')

    # softmax
    # x = x.to(Softmax(), name='loss')


    # x.to(Softmax(), name='prob')
    mod.vars(x, 'label', 'label_weight').to(SoftmaxWithLoss(axis=2), name='loss')
    # ?when keeping accuracy, it's wrong
    # mod.vars(x, 'label').to(Accuracy(1), name='accuracy_top1')
    # mod.vars(x, 'label').to(Accuracy(5), name='accuracy_top5')
    model = mod.compile(inputs=inputs, seal=False)
    model.add_flow('main',
                   inputs.keys(), ['loss', 'accuracy_top1', 'accuracy_top5'],
                   ['loss'])
    model.seal()
    return model


if __name__ == '__main__':
    base.set_debug_log(True)
    model = create_model()
    with open('params.yaml', 'w+') as f:
        f.write(model.to_yaml_text())
