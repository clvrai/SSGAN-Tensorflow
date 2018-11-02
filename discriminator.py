#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from ops import conv2d
from util import log


class Discriminator(object):
    def __init__(self, name, num_class, norm_type, is_train):
        self.name = name
        self._num_class = num_class
        self._norm_type = norm_type
        self._is_train = is_train
        self._reuse = False

    def __call__(self, input):
        with tf.variable_scope(self.name, reuse=self._reuse):
            if not self._reuse:
                print('\033[93m'+self.name+'\033[0m')
            _ = input
            num_channel = [32, 64, 128, 256, 256, 512]
            num_layer = np.ceil(np.log2(min(_.shape.as_list()[1:3]))).astype(np.int)
            for i in range(num_layer):
                ch = num_channel[i] if i < len(num_channel) else 512
                _ = conv2d(_, ch, self._is_train, info=not self._reuse,
                           norm=self._norm_type, name='conv{}'.format(i+1))
            _ = conv2d(_, int(num_channel[i]/4), self._is_train, k=1, s=1,
                       info=not self._reuse, norm='None', name='conv{}'.format(i+2))
            _ = conv2d(_, self._num_class+1, self._is_train, k=1, s=1, info=not self._reuse,
                       activation_fn=None, norm='None',
                       name='conv{}'.format(i+3))
            _ = tf.squeeze(_)
            if not self._reuse: 
                log.info('discriminator output {}'.format(_.shape.as_list()))
            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return tf.nn.sigmoid(_), _
