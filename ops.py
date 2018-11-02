import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
import numpy as np
from util import log

def print_info(name, shape, activation_fn):
    log.info('{}{} {}'.format(
        name,  '' if activation_fn is None else ' ('+activation_fn.__name__+')',
        shape))


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)


def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


def instance_norm(input):
    """
    Instance normalization
    """
    with tf.variable_scope('instance_norm'):
        num_out = input.get_shape()[-1]
        scale = tf.get_variable(
            'scale', [num_out],
            initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable(
            'offset', [num_out],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
        mean, var = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-6
        inv = tf.rsqrt(var + epsilon)
        return scale * (input - mean) * inv + offset


def norm_and_act(input, is_train, norm='batch', activation_fn=None, name="bn_act"):
    """
    Apply normalization and/or activation function
    """
    with tf.variable_scope(name):
        _ = input
        if activation_fn is not None:
            _ = activation_fn(_)
        if norm is not None and norm is not False:
            if norm == 'batch':
                _ = tf.contrib.layers.batch_norm(
                    _, center=True, scale=True,
                    updates_collections=None,
                )
            elif norm == 'instance':
                _ = instance_norm(_, is_train)
            elif norm == 'None':
                _ = _
            else:
                raise NotImplementedError
    return _


def conv2d(input, output_shape, is_train, info=False, k=3, s=2, stddev=0.01, 
           activation_fn=lrelu, norm='batch', name="conv2d"):
    with tf.variable_scope(name):
        _ = slim.conv2d(input, output_shape, [k, k], stride=s, activation_fn=None)
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _


def deconv2d(input, output_shape, is_train, info=False, k=3, s=2, stddev=0.01, 
             activation_fn=tf.nn.relu, norm='batch', name='deconv2d'):
    with tf.variable_scope(name):
        _ = layers.conv2d_transpose(
            input,
            num_outputs=output_shape,
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            biases_initializer=tf.zeros_initializer(),
            activation_fn=None,
            kernel_size=[k, k], stride=[s, s], padding='SAME'
        )
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _


def bilinear_deconv2d(input, output_shape, is_train, info=False, k=3, s=2, stddev=0.01, 
                      activation_fn=tf.nn.relu, norm='batch', name='deconv2d'):
    with tf.variable_scope(name):
        h = int(input.get_shape()[1]) * s
        w = int(input.get_shape()[2]) * s
        _ = tf.image.resize_bilinear(input, [h, w])
        _ = conv2d(_, output_shape, is_train, k=k, s=1,
                   norm=False, activation_fn=None)
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _


def nn_deconv2d(input, output_shape, is_train, info=False, k=3, s=2, stddev=0.01, 
                activation_fn=tf.nn.relu, norm='batch', name='deconv2d'):
    with tf.variable_scope(name):
        h = int(input.get_shape()[1]) * s
        w = int(input.get_shape()[2]) * s
        _ = tf.image.resize_nearest_neighbor(input, [h, w])
        _ = conv2d(_, output_shape, is_train, k=k, s=1,
                   norm=False, activation_fn=None)
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _


def fc(input, output_shape, is_train, info=False, norm='batch',
       activation_fn=lrelu, name="fc"):
    with tf.variable_scope(name):
        _ = slim.fully_connected(input, output_shape, activation_fn=None)
        _ = norm_and_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: print_info(name, _.get_shape().as_list(), activation_fn)
    return _
