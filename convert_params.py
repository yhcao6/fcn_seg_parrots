import h5py
import numpy as np
import caffe

caffe.set_mode_gpu()
caffe.set_device(0)

f = h5py.File('./vgg16.parrots')
# first layer
f['conv1_1.b@value'] = np.reshape(f.pop('conv1_1.b@value'), [1, 64, 1, 1] ,order='F')
f['conv1_2.b@value'] = np.reshape(f.pop('conv1_2.b@value'), [1, 64, 1, 1] ,order='F')

# second layer
f['conv2_1.b@value'] = np.reshape(f.pop('conv2_1.b@value'), [1, 128, 1, 1], order='F')
f['conv2_2.b@value'] = np.reshape(f.pop('conv2_2.b@value'), [1, 128, 1, 1], order='F')

# third layer
f['conv3_1.b@value'] = np.reshape(f.pop('conv3_1.b@value'), [1, 256, 1, 1], order='F')
f['conv3_2.b@value'] = np.reshape(f.pop('conv3_2.b@value'), [1, 256, 1, 1], order='F')
f['conv3_3.b@value'] = np.reshape(f.pop('conv3_3.b@value'), [1, 256, 1, 1], order='F')

# forth layer
f['conv4_1.b@value'] = np.reshape(f.pop('conv4_1.b@value'), [1, 512, 1, 1], order='F')
f['conv4_2.b@value'] = np.reshape(f.pop('conv4_2.b@value'), [1, 512, 1, 1], order='F')
f['conv4_3.b@value'] = np.reshape(f.pop('conv4_3.b@value'), [1, 512, 1, 1], order='F')

# fifth layer
f['conv5_1.b@value'] = np.reshape(f.pop('conv5_1.b@value'), [1, 512, 1, 1], order='F')
f['conv5_2.b@value'] = np.reshape(f.pop('conv5_2.b@value'), [1, 512, 1, 1], order='F')
f['conv5_3.b@value'] = np.reshape(f.pop('conv5_3.b@value'), [1, 512, 1, 1], order='F')

# fc6
f['conv_fc6.w@value'] = np.reshape(f.pop('fc6.w@value'), [4096, 512, 7, 7], order='F')
f['conv_fc6.b@value'] = np.reshape(f.pop('fc6.b@value'), [1, 4096, 1, 1], order='F')

# fc7
f['conv_fc7.w@value'] = np.reshape(f.pop('fc7.w@value'), [4096, 4096, 1, 1], order='F')
f['conv_fc7.b@value'] = np.reshape(f.pop('fc7.b@value'), [1, 4096, 1, 1], order='F')

MODEL = '/home/yhcao/VGG16_seg/trainval.prototxt'
WEIGHT = '/home/yhcao/VGG16_seg/VGG_16.caffemodel'
net = caffe.Net(MODEL, WEIGHT, caffe.TRAIN)

# fc8
f['conv_fc8.w@value'] = net.params['fc8-VOC'][0].data
f['conv_fc8.b@value'] = net.params['fc8-VOC'][1].data

# upscore2
tmp = np.zeros((21, 21, 16, 16))
caffe_upscore2 = net.params['upscore2'][0].data
for i in range(21):
    tmp[i, i, ...] = caffe_upscore2[i, 0, ...]
f['upscore2.w@value'] = tmp


