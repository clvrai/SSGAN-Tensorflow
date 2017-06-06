from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim

from ops import *
from util import log

class Model(object):

    def __init__(self, config,
                 debug_information=False,
                 is_train=True):
        self.debug = debug_information

        self.config = config
        self.batch_size = self.config.batch_size
        self.input_height = self.config.data_info[0]
        self.input_width = self.config.data_info[1]
        self.num_class = self.config.data_info[2]
        self.c_dim = self.config.data_info[3]
        self.deconv_info = self.config.deconv_info
        self.conv_info = self.config.conv_info

        # create placeholders for the input
        self.image = tf.placeholder(
            name='image', dtype=tf.float32,
            shape=[self.batch_size, self.input_height, self.input_width, self.c_dim],
        )
        self.label = tf.placeholder(
            name='label', dtype=tf.float32, shape=[self.batch_size, self.num_class],
        )

        self.is_training = tf.placeholder_with_default(bool(is_train), [], name='is_training')

        self.build(is_train=is_train)

    def get_feed_dict(self, batch_chunk, step=None, is_training=None):
        fd = {
            self.image: batch_chunk['image'], # [B, h, w, c]
            self.label: batch_chunk['label'], # [B, n]
        }
        if is_training is not None:
            fd[self.is_training] = is_training
        return fd

    def build(self, is_train=True):

        n = self.num_class
        h = self.input_height
        w = self.input_width
        deconv_info = self.deconv_info
        conv_info = self.conv_info
        c_dim = self.c_dim
        n_z = 100

        # G takes ramdon noise and tries to generate images [B, h, w, c]
        def G(z, scope='Generator'):
            with tf.variable_scope(scope) as scope:
                print ('\033[93m'+scope.name+'\033[0m')
                z = tf.reshape(z, [self.batch_size, 1, 1, -1])
                g_1 = deconv2d(z, deconv_info[0], is_train, name='g_1_deconv') 
                print (scope.name, g_1)
                g_2 = deconv2d(g_1, deconv_info[1], is_train, name='g_2_deconv')
                print (scope.name, g_2)
                g_3 = deconv2d(g_2, deconv_info[2], is_train, name='g_3_deconv')
                print (scope.name, g_3)
                g_4 = deconv2d(g_3, deconv_info[3], is_train, name='g_4_deconv', activation_fn='tanh')
                print (scope.name, g_4)
                output = g_4
                assert output.get_shape().as_list() == self.image.get_shape().as_list(), output.get_shape().as_list()
            return output

        # D takes images as input and tries to output class label [B, n+1]
        def D(img, scope='Discriminator', reuse=True):
            with tf.variable_scope(scope, reuse=reuse) as scope:
                if not reuse: print ('\033[93m'+scope.name+'\033[0m')
                d_1 = conv2d(img, conv_info[0], is_train, name='d_1_conv')
                d_1 = slim.dropout(d_1, keep_prob=0.5, is_training=is_train, scope='d_1_conv/')
                if not reuse: print (scope.name, d_1)
                d_2 = conv2d(d_1, conv_info[1], is_train, name='d_2_conv')
                d_2 = slim.dropout(d_2, keep_prob=0.5, is_training=is_train, scope='d_2_conv/')
                if not reuse: print (scope.name, d_2)
                d_3 = conv2d(d_2, conv_info[2], is_train, name='d_3_conv')
                d_3 = slim.dropout(d_3, keep_prob=0.5, is_training=is_train, scope='d_3_conv/')
                if not reuse: print (scope.name, d_3)
                d_4 = slim.fully_connected(
                    tf.reshape(d_3, [self.batch_size, -1]), n+1, scope='d_4_fc', activation_fn=None)
                if not reuse: print (scope.name, d_4)
                output = d_4
                assert output.get_shape().as_list() == [self.batch_size, n+1]
                return tf.nn.sigmoid(output), output

        # Generator {{{
        # =========
        z = tf.random_uniform([self.batch_size, n_z], minval=-1, maxval=1, dtype=tf.float32)
        fake_image = G(z)
        self.fake_img = fake_image
        # }}}

        # Discriminator {{{
        # =========
        d_real, d_real_logits = D(self.image, scope='Discriminator', reuse=False)
        d_fake, d_fake_logits = D(fake_image, scope='Discriminator', reuse=True)
        self.all_preds = d_real
        self.all_targets = self.label
        # }}}

        # build loss and self.accuracy{{{
        # Supervised loss
        # cross-entropy
        self.S_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                     logits=d_real_logits[:, :-1], labels=self.label))

        # GAN loss
        alpha = 0.9
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                     logits=d_real_logits[:, -1], labels=tf.zeros_like(d_real[:, -1])))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                     logits=d_fake_logits[:, -1], labels=alpha*tf.ones_like(d_fake[:, -1], dtype=tf.float32)))
        self.d_loss = d_loss_real + d_loss_fake + self.S_loss
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                     logits=d_fake_logits[:, -1], labels=tf.zeros_like(d_fake[:, -1])))
        GAN_loss = tf.reduce_mean(self.d_loss + self.g_loss)

        # Classification accuracy
        correct_prediction = tf.equal(tf.argmax(d_real_logits[:, :-1], 1), tf.argmax(self.label,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # }}}

        tf.summary.scalar("loss/accuracy", self.accuracy)
        tf.summary.scalar("loss/GAN_loss", GAN_loss)
        tf.summary.scalar("loss/S_loss", self.S_loss)
        tf.summary.scalar("loss/d_loss", tf.reduce_mean(self.d_loss))
        tf.summary.scalar("loss/d_loss_real", tf.reduce_mean(d_loss_real))
        tf.summary.scalar("loss/d_loss_fake", tf.reduce_mean(d_loss_fake))
        tf.summary.scalar("loss/g_loss", tf.reduce_mean(self.g_loss))
        tf.summary.image("img/fake", fake_image)
        tf.summary.image("img/real", self.image, max_outputs=1)
        tf.summary.image("label/target_real", tf.reshape(self.label, [1, self.batch_size, n, 1]))
        tf.summary.image("label/pred_real", tf.reshape(d_real, [1, self.batch_size, n+1, 1]))
        tf.summary.image("label/pred_fake", tf.reshape(d_fake, [1, self.batch_size, n+1, 1]))
        print ('\033[93mSuccessfully loaded the model.\033[0m')
