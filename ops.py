import tensorflow as tf
import tensorflow.contrib.layers as layers

def lrelu(x, leak=0.1, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)

def conv2d(input, output_shape, is_train, k_h=5, k_w=5, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input.get_shape()[-1], output_shape],
                initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input, w, strides=[1, 2, 2, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_shape], initializer=tf.constant_initializer(0.0))
        conv = lrelu(tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape()))
        bn = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9, is_training=is_train, updates_collections=None)
    return bn

def deconv2d(input, deconv_info, name="deconv2d", activation_fn='lrelu'):
    with tf.variable_scope(name):
        output_shape = deconv_info[0]
        k = deconv_info[1]
        s = deconv_info[2]
        deconv = layers.conv2d_transpose(input,
            num_outputs=output_shape,
            kernel_size=[k, k], stride=[s, s], padding='VALID')
        if activation_fn == 'lrelu':
            deconv = lrelu(deconv)
        elif activation_fn == 'tanh':
            deconv = tf.nn.tanh(deconv)
        return deconv
