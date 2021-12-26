# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division

import tensorflow.compat.v1 as tf
#import tensorflow.contrib.slim as slim
import tf_slim as slim


@slim.add_arg_scope
def conv2d_fixed(inputs, num_outputs, kernel_size, **kwargs):
    return slim.conv2d(inputs, num_outputs, kernel_size, **kwargs)


def spatiotemporal_arg_scope(is_training=True,
                      weight_decay=0.0001,
                      batch_norm_decay=0.95,
                      batch_norm_epsilon=1e-5,
                      batch_norm_scale=True):
    """Defines the spatiotemporal arg scope.

    Args:
        weight_decay: The l2 regularization coefficient.

    Returns:
        An arg_scope.
    """
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    batch_norm_params_fixed = {
        'is_training': False,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': False,
    }

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, conv2d_fixed], padding='SAME',
                        activation_fn=tf.nn.relu,
                        biases_initializer=tf.constant_initializer(0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([conv2d_fixed], normalizer_params=batch_norm_params_fixed, trainable=False):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


def spatiotemporal(input1, input2, scope='spatiotemporal'):
    """
    Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        scope: Optional scope for the variables.

    Returns:
        the last op containing the log predictions.
    """
    with tf.variable_scope(scope, 'spatiotemporal', [input1, input2]):
        with tf.variable_scope('jointDis', [input1, input2]):
            conv1_a = conv2d_fixed(input1, 16, 3, scope='conv1')
            conv1_b = conv2d_fixed(input2, 16, 3, scope='conv1', reuse=True)
            conv2_a = conv2d_fixed(conv1_a, 16, 3, scope='conv2')
            conv2_b = conv2d_fixed(conv1_b, 16, 3, scope='conv2', reuse=True)
            norm1_a = tf.nn.lrn(conv2_a, 5, 2e-5, 0.0001, 0.75,
                                            'norm1_a')
            norm1_b = tf.nn.lrn(conv2_b, 5, 2e-5, 0.0001, 0.75,
                                            'norm1_b')
            pool1_a = slim.max_pool2d(norm1_a, 3, scope='pool1_a')
            pool1_b = slim.max_pool2d(norm1_b, 3, scope='pool1_b')
            conv3_a = conv2d_fixed(pool1_a, 32, 3, scope='conv3')
            conv3_b = conv2d_fixed(pool1_b, 32, 3, scope='conv3', reuse=True)
            pool2_a = slim.max_pool2d(conv3_a, 3, scope='pool2_a')
            pool2_b = slim.max_pool2d(conv3_b, 3, scope='pool2_b')
            conv4_a = conv2d_fixed(pool2_a, 64, 3, scope='conv4')
            conv4_b = conv2d_fixed(pool2_b, 64, 3, scope='conv4', reuse=True)
            conv5_S = conv2d_fixed(conv4_a, 64, 3, scope='conv5_S')
            conv5_T = conv2d_fixed(tf.concat([conv4_a, tf.subtract(conv4_a, conv4_b)],
                                    -1), 64, 3, scope='conv5_T')
            conv6_S = conv2d_fixed(conv5_S, 128, 3, scope='conv6_S')
            conv6_T = conv2d_fixed(conv5_T, 128, 3, scope='conv6_T')

        conv7_ST = slim.conv2d(tf.concat([conv6_S, conv6_T], -1), 32, 1,
                               scope='conv7_ST')
        conv8_ST = slim.conv2d(conv7_ST, 32, 3, scope='conv8_ST')
        conv9_ST = slim.conv2d(conv8_ST, 32, 3, scope='conv9_ST')
        conv10_ST = slim.conv2d(conv9_ST, 16, 3, scope='conv10_ST')
        deconv1_ST = slim.conv2d_transpose(conv10_ST, 8, 4, stride=2, activation_fn=None,
                                      scope='deconv1_ST')
        deconv2_ST = slim.conv2d_transpose(deconv1_ST, 1, 4, stride=2, activation_fn=None,
                                      scope='deconv2_ST')

    return deconv2_ST
