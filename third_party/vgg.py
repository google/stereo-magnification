# ******************************************************************************
# VGG network definition from:
#
#  https://github.com/CQFIO/PhotographicImageSynthesis/blob/master/demo_1024p.py
#
# Released under an MIT License.
"""VGG network definition.
"""

from __future__ import division
import numpy as np
import tensorflow as tf


def build_net(ntype, nin, nwb=None, name=None):
  if ntype == 'conv':
    return tf.nn.relu(
        tf.nn.conv2d(
            nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME', name=name) +
        nwb[1])
  elif ntype == 'pool':
    return tf.nn.avg_pool(
        nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def get_weight_bias(vgg_layers, i):
  weights = vgg_layers[i][0][0][2][0][0]
  weights = tf.constant(weights)
  bias = vgg_layers[i][0][0][2][0][1]
  bias = tf.constant(np.reshape(bias, (bias.size)))
  return weights, bias


def build_vgg19(input, model_filepath, reuse=False):
  with tf.variable_scope('vgg', reuse=reuse):
    net = {}
    input = tf.cast(input, tf.float32)
    import scipy.io as sio
    with open(model_filepath, 'r') as f:
      vgg_rawnet = sio.loadmat(f)
    vgg_layers = vgg_rawnet['layers'][0]
    imagenet_mean = tf.constant(
        [123.6800, 116.7790, 103.9390], shape=[1, 1, 1, 3])
    net['input'] = input - imagenet_mean
    net['conv1_1'] = build_net(
        'conv',
        net['input'],
        get_weight_bias(vgg_layers, 0),
        name='vgg_conv1_1')
    net['conv1_2'] = build_net(
        'conv',
        net['conv1_1'],
        get_weight_bias(vgg_layers, 2),
        name='vgg_conv1_2')
    net['pool1'] = build_net('pool', net['conv1_2'])
    net['conv2_1'] = build_net(
        'conv',
        net['pool1'],
        get_weight_bias(vgg_layers, 5),
        name='vgg_conv2_1')
    net['conv2_2'] = build_net(
        'conv',
        net['conv2_1'],
        get_weight_bias(vgg_layers, 7),
        name='vgg_conv2_2')
    net['pool2'] = build_net('pool', net['conv2_2'])
    net['conv3_1'] = build_net(
        'conv',
        net['pool2'],
        get_weight_bias(vgg_layers, 10),
        name='vgg_conv3_1')
    net['conv3_2'] = build_net(
        'conv',
        net['conv3_1'],
        get_weight_bias(vgg_layers, 12),
        name='vgg_conv3_2')
    net['conv3_3'] = build_net(
        'conv',
        net['conv3_2'],
        get_weight_bias(vgg_layers, 14),
        name='vgg_conv3_3')
    net['conv3_4'] = build_net(
        'conv',
        net['conv3_3'],
        get_weight_bias(vgg_layers, 16),
        name='vgg_conv3_4')
    net['pool3'] = build_net('pool', net['conv3_4'])
    net['conv4_1'] = build_net(
        'conv',
        net['pool3'],
        get_weight_bias(vgg_layers, 19),
        name='vgg_conv4_1')
    net['conv4_2'] = build_net(
        'conv',
        net['conv4_1'],
        get_weight_bias(vgg_layers, 21),
        name='vgg_conv4_2')
    net['conv4_3'] = build_net(
        'conv',
        net['conv4_2'],
        get_weight_bias(vgg_layers, 23),
        name='vgg_conv4_3')
    net['conv4_4'] = build_net(
        'conv',
        net['conv4_3'],
        get_weight_bias(vgg_layers, 25),
        name='vgg_conv4_4')
    net['pool4'] = build_net('pool', net['conv4_4'])
    net['conv5_1'] = build_net(
        'conv',
        net['pool4'],
        get_weight_bias(vgg_layers, 28),
        name='vgg_conv5_1')
    net['conv5_2'] = build_net(
        'conv',
        net['conv5_1'],
        get_weight_bias(vgg_layers, 30),
        name='vgg_conv5_2')
  return net

# ******************************************************************************
