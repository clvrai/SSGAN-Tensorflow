from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from ops import huber_loss
from util import log
from generator import Generator
from discriminator import Discriminator


class Model(object):

    def __init__(self, config,
                 debug_information=False,
                 is_train=True):
        self.debug = debug_information

        self.config = config
        self.batch_size = self.config.batch_size
        self.h = self.config.h
        self.w = self.config.w
        self.c = self.config.c
        self.num_class = self.config.num_class
        self.n_z = config.n_z
        self.norm_type = config.norm_type
        self.deconv_type = config.deconv_type

        # create placeholders for the input
        self.image = tf.placeholder(
            name='image', dtype=tf.float32,
            shape=[self.batch_size, self.h, self.w, self.c],
        )
        self.label = tf.placeholder(
            name='label', dtype=tf.float32, shape=[self.batch_size, self.num_class],
        )

        self.is_training = tf.placeholder_with_default(bool(is_train), [], name='is_training')

        self.recon_weight = tf.placeholder_with_default(
            tf.cast(1.0, tf.float32), [])
        tf.summary.scalar("loss/recon_wieght", self.recon_weight)

        self.build(is_train=is_train)

    def get_feed_dict(self, batch_chunk, step=None, is_training=None):
        fd = {
            self.image: batch_chunk['image'],  # [bs, h, w, c]
            self.label: batch_chunk['label'],  # [bs, n]
        }
        if is_training is not None:
            fd[self.is_training] = is_training

        # Weight annealing
        if step is not None:
            fd[self.recon_weight] = min(max(0, (1500 - step) / 1500), 1.0)*10
        return fd

    def build(self, is_train=True):

        n = self.num_class

        # build loss and accuracy {{{
        def build_loss(d_real, d_real_logits, d_fake, d_fake_logits, label, real_image, fake_image):
            alpha = 0.9
            real_label = tf.concat([label, tf.zeros([self.batch_size, 1])], axis=1)
            fake_label = tf.concat([(1-alpha)*tf.ones([self.batch_size, n])/n,
                                    alpha*tf.ones([self.batch_size, 1])], axis=1)

            # Discriminator/classifier loss
            s_loss = tf.reduce_mean(huber_loss(label, d_real[:, :-1]))
            d_loss_real = tf.nn.softmax_cross_entropy_with_logits(
                logits=d_real_logits, labels=real_label)
            d_loss_fake = tf.nn.softmax_cross_entropy_with_logits(
                logits=d_fake_logits, labels=fake_label)
            d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)

            # Generator loss
            g_loss = tf.reduce_mean(tf.log(d_fake[:, -1]))

            # Weight annealing
            g_loss += tf.reduce_mean(
                huber_loss(real_image, fake_image)) * self.recon_weight

            GAN_loss = tf.reduce_mean(d_loss + g_loss)

            # Classification accuracy
            correct_prediction = tf.equal(tf.argmax(d_real[:, :-1], 1),
                                          tf.argmax(self.label, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return s_loss, d_loss_real, d_loss_fake, d_loss, g_loss, GAN_loss, accuracy
        # }}}

        # Generator {{{
        # =========
        G = Generator('Generator', self.h, self.w, self.c,
                      self.norm_type, self.deconv_type, is_train)
        z = tf.random_uniform([self.batch_size, self.n_z],
                              minval=-1, maxval=1, dtype=tf.float32)
        fake_image = G(z)
        self.fake_image = fake_image
        # }}}

        # Discriminator {{{
        # =========
        D = Discriminator('Discriminator', self.num_class, self.norm_type, is_train)
        d_real, d_real_logits = D(self.image)
        d_fake, d_fake_logits = D(fake_image)
        self.all_preds = d_real
        self.all_targets = self.label
        # }}}

        self.S_loss, d_loss_real, d_loss_fake, self.d_loss, self.g_loss, GAN_loss, self.accuracy = \
            build_loss(d_real, d_real_logits, d_fake, d_fake_logits, self.label, self.image, fake_image)

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
        log.warn('\033[93mSuccessfully loaded the model.\033[0m')
