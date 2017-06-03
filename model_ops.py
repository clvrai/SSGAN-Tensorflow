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


def CONV(r, num_outputs, scope, reuse=True, is_train=True):
    # input: [B, 90, 160, 25]
    # output: [B, num_cell_units]
    with tf.variable_scope(scope, reuse=reuse):
        if reuse is False: print ('\033[93m'+scope+'\033[0m')
        # FIXME w/o frames
        feature_map_num = [1, 32, 32, 64, 64, 128, 128]  #set0
        with tf.variable_scope('conv_2d_1') as scope:
            kernel = tf.get_variable('weights', [7, 7, feature_map_num[0], feature_map_num[1]],
                                     dtype=tf.float32, initializer=layers.xavier_initializer_conv2d())
            conv = tf.nn.conv2d(r, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [feature_map_num[1]])
            conv1 = tf.nn.bias_add(conv, biases)
            relu1 = lrelu(conv1)
            bn1 = tf.contrib.layers.batch_norm(relu1, center=True, scale=True, decay=0.9, is_training=is_train, updates_collections=None)
            pool1 = tf.nn.max_pool(bn1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
            if reuse is False: print (scope.name, pool1.get_shape())
        # [B, 45, 80, 64]
        with tf.variable_scope('conv_2d_2') as scope:
            kernel = tf.get_variable('weights', [5, 5, feature_map_num[1], feature_map_num[2]],
                                        dtype=tf.float32, initializer=layers.xavier_initializer_conv2d())
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [feature_map_num[2]])
            conv2 = tf.nn.bias_add(conv, biases)
            relu2 = lrelu(conv2)
            bn2 = tf.contrib.layers.batch_norm(relu2, center=True, scale=True, decay=0.9, is_training=is_train, updates_collections=None)
            pool2 = tf.nn.max_pool(bn2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
            if reuse is False: print (scope.name, pool2.get_shape())
        # [B, 22, 40, 128]
        with tf.variable_scope('conv_2d_3') as scope:
            kernel = tf.get_variable('weights', [5, 5, feature_map_num[2], feature_map_num[3]],
                                        dtype=tf.float32, initializer=layers.xavier_initializer_conv2d())
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [feature_map_num[3]])
            conv3 = tf.nn.bias_add(conv, biases)
            relu3 = lrelu(conv3)
            bn3 = tf.contrib.layers.batch_norm(relu3, center=True, scale=True, decay=0.9, is_training=is_train, updates_collections=None)
            pool3 = tf.nn.max_pool(bn3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
            if reuse is False: print (scope.name, pool3.get_shape())
        # [B, 11, 20, 256]
        with tf.variable_scope('conv_2d_4') as scope:
            kernel = tf.get_variable('weights', [3, 3, feature_map_num[3], feature_map_num[4]],
                                        dtype=tf.float32, initializer=layers.xavier_initializer_conv2d())
            conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [feature_map_num[4]])
            conv4 = tf.nn.bias_add(conv, biases)
            relu4 = lrelu(conv4)
            bn4 = tf.contrib.layers.batch_norm(relu4, center=True, scale=True, decay=0.9, is_training=is_train, updates_collections=None)
            pool4 = tf.nn.max_pool(bn4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
            if reuse is False: print (scope.name, pool4.get_shape())
        # [B, 6, 10, 512]
        with tf.variable_scope('conv_2d_5') as scope:
            kernel = tf.get_variable('weights', [3, 3, feature_map_num[4], feature_map_num[5]],
                                        dtype=tf.float32, initializer=layers.xavier_initializer_conv2d())
            conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [feature_map_num[5]])
            conv5 = tf.nn.bias_add(conv, biases)
            relu5 = lrelu(conv5)
            bn5 = tf.contrib.layers.batch_norm(relu5, center=True, scale=True, decay=0.9, is_training=is_train, updates_collections=None)
            pool5 = tf.nn.max_pool(bn5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
            if reuse is False: print (scope.name, pool5.get_shape())
        # [B, 3, 5, 512]
        with tf.variable_scope('conv_2d_6') as scope:
            kernel = tf.get_variable('weights', [3, 3, feature_map_num[5], feature_map_num[6]],
                                        dtype=tf.float32, initializer=layers.xavier_initializer_conv2d())
            conv = tf.nn.conv2d(pool5, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', [feature_map_num[6]])
            conv6 = tf.nn.bias_add(conv, biases)
            relu6 = lrelu(conv6)
            bn6 = tf.contrib.layers.batch_norm(relu6, center=True, scale=True, decay=0.9, is_training=is_train, updates_collections=None)
            pool6 = tf.nn.max_pool(bn6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool6')
            if reuse is False: print (scope.name, pool6.get_shape())

        with tf.variable_scope('conv_fc_1') as scope:
            conv_fc_1 = lrelu(tf.contrib.layers.fully_connected(
                inputs=pool6,
                num_outputs=num_outputs*2,
                activation_fn=lrelu,
                weights_initializer=tf.contrib.layers.xavier_initializer()
            ))
        if reuse is False: print (scope.name, conv_fc_1.get_shape())
        with tf.variable_scope('conv_fc') as scope:
            conv_fc = lrelu(tf.contrib.layers.fully_connected(
                inputs=conv_fc_1,
                num_outputs=num_outputs,
                # activation_fn=tf.nn.relu, #tf.nn.tanh,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                scope='conv_fc',
            ))
        if reuse is False: print (scope.name, conv_fc.get_shape())
        # [B, num_cell_units]
        return conv_fc #
