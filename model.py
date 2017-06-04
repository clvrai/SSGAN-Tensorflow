from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim

from model_ops import lrelu, huber_loss

from util import log

class Model(object):

    def __init__(self, config,
                 debug_information=False,
                 is_train=True):
        self.debug = debug_information

        self.config = config
        self.batch_size = self.config.batch_size
        self.input_height = self.config.input_height
        self.input_width = self.config.input_width
        self.num_class = self.config.num_class

        # create placeholders for the input
        self.image = tf.placeholder(
            name='image', dtype=tf.float32,
            shape=[self.batch_size, self.input_height, self.input_width],
        )
        self.label = tf.placeholder(
            name='label', dtype=tf.float32, shape=[self.batch_size, self.num_class],
        )

        self.is_training = tf.placeholder_with_default(bool(is_train), [], name='is_training')

        self.build(is_train=is_train)

    def get_feed_dict(self, batch_chunk, step=None, is_training=None):
        fd = {
            self.image: batch_chunk['image'], # [B, h, w]
            self.label: batch_chunk['label'], # [B, n]
        }
        if is_training is not None:
            fd[self.is_training] = is_training
        return fd

    def build(self, is_train=True):

        n = self.num_class
        h = self.input_height
        w = self.input_width
        n_z = 100

        # XXX G takes ramdon noise and tries to generate images [B, h, w]
        def G(z, scope='Generator'):
            with tf.variable_scope(scope) as scope:
                print ('\033[93m'+scope.name+'\033[0m')
                with slim.arg_scope([slim.fully_connected],
                                    activation_fn=lrelu, weights_initializer=layers.xavier_initializer()):
                    x_1 = slim.fully_connected(z, 200, scope='x_1')
                    x_1 = slim.dropout(x_1, keep_prob=0.5, is_training=is_train, scope='x_1/')
                    print(scope.name, x_1)
                    x_2 = slim.fully_connected(x_1, 400, scope='x_2')
                    x_2 = slim.dropout(x_2, keep_prob=0.5, is_training=is_train, scope='x_2/')
                    print(scope.name, x_2)
                    x_3 = slim.fully_connected(x_2, 800, scope='x_3',
                                               activation_fn=lrelu)
                    x_3 = slim.dropout(x_3, keep_prob=0.5, is_training=is_train, scope='x_3/')
                    print(scope.name, x_3)
                    x_4 = slim.fully_connected(x_3, h*w, scope='x_4',
                                               activation_fn=tf.sigmoid )
                    print (scope.name, x_4)
                    output = tf.reshape(x_4, shape=[self.batch_size, h, w])
                    assert output.get_shape().as_list() == [self.batch_size, h, w]
            return output

        # XXX D takes images as input and tries to output class label [B, n+1]
        def D(img, scope='Discriminator', reuse=True):
            with tf.variable_scope(scope, reuse=reuse) as scope:
                if not reuse: print ('\033[93m'+scope.name+'\033[0m')
                img = tf.reshape(img, [self.batch_size, -1])
                with slim.arg_scope([slim.fully_connected],
                                    activation_fn=lrelu, weights_initializer=layers.xavier_initializer()):
                    d_1 = slim.fully_connected(img, 400, scope='d_1')
                    d_1 = tf.contrib.layers.dropout(d_1, keep_prob=0.5, is_training=is_train, scope='d_1/')
                    if not reuse: print (scope.name, d_1)
                    d_2 = slim.fully_connected(d_1, 100, scope='d_2')
                    d_2 = tf.contrib.layers.dropout(d_2, keep_prob=0.5, is_training=is_train, scope='d_2/')
                    if not reuse: print (scope.name, d_2)
                    d_3 = slim.fully_connected(d_2, 25, scope='d_3')
                    d_3 = tf.contrib.layers.dropout(d_3, keep_prob=0.5, is_training=is_train, scope='d_3/')
                    if not reuse: print (scope.name, d_3)
                    d_4 = slim.fully_connected(d_3, n+1, scope='d_4',
                                               activation_fn=None)
                    if not reuse: print (scope.name, d_4.get_shape())
                    output = d_4
                    assert output.get_shape().as_list() == [self.batch_size, n+1]
                return tf.nn.sigmoid(output), output

        # Generator {{{
        # =========
        z = tf.random_normal(shape=[self.batch_size, n_z])
        self.z = z
        fake_image = G(z)
        assert self.image.get_shape().as_list() == fake_image.get_shape().as_list(), fake_image.get_shape().as_list()
        # }}}

        # Discriminator {{{
        # =========
        d_real, d_real_logits = D(self.image, scope='Discriminator', reuse=False)
        self.all_preds = d_real
        self.all_targets = self.label
        d_fake, d_fake_logits = D(fake_image, scope='Discriminator', reuse=True)
        d_real_logits.get_shape().assert_is_compatible_with([self.batch_size, n+1])
        d_fake_logits.get_shape().assert_is_compatible_with([self.batch_size, n+1])
        # }}}

        # build loss {{{
        # XXX supervised loss
        # cross-entropy
        self.S_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits[:, :-1], labels=self.label))

        # XXX GAN loss
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits[:, -1], labels=tf.ones_like(d_real[:, -1])))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits[:, -1], labels=tf.zeros_like(d_fake[:, -1])))
        self.d_loss = d_loss_real + d_loss_fake
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits[:, -1], labels=tf.ones_like(d_fake[:, -1])))
        GAN_loss = tf.reduce_mean(self.d_loss + self.g_loss)

        # Total loss
        self.total_loss = tf.reduce_mean( self.S_loss + GAN_loss )

        # }}}

        tf.summary.scalar("loss/total_loss", self.total_loss)
        tf.summary.scalar("loss/GAN_loss", GAN_loss)
        tf.summary.scalar("loss/S_loss", self.S_loss)
        tf.summary.scalar("loss/d_loss", tf.reduce_mean(self.d_loss))
        tf.summary.scalar("loss/d_loss_real", tf.reduce_mean(d_loss_real))
        tf.summary.scalar("loss/d_loss_fake", tf.reduce_mean(d_loss_fake))
        tf.summary.scalar("loss/g_loss", tf.reduce_mean(self.g_loss))
        for i in range(min(4, self.batch_size)):
            tf.summary.image("generated_img/"+str(i), 
                tf.expand_dims(tf.expand_dims(
                    tf.reshape(fake_image[i, :], shape=[h, w]), dim=0), dim=-1))
        tf.summary.image("real_img/0", 
            tf.expand_dims(tf.expand_dims(
                tf.reshape(self.image[0, :], shape=[h, w]), dim=0), dim=-1))
        print ('\033[93mSuccessfully loaded the model.\033[0m')
