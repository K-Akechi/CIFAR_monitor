import tensorflow as tf


def weight_variable(shape):
    # initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable('weights', dtype='float32', shape=shape, initializer=tf.contrib.keras.initializers.he_normal())


def bias_variable(shape):
    # initial = tf.constant(0.1, shape=shape, dtype=float)
    return tf.get_variable('bias', shape=shape, dtype='float32', initializer=tf.constant_initializer(0.1))


def conv2d(x, W, stride=1):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def batch_norm(x, train_flag):
    return tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True, epsilon=1e-3, is_training=train_flag,
                                        updates_collections=None)


def global_avg_pool(x, k_size=1, stride=1):
    return tf.nn.avg_pool(x, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1], padding='VALID')


def residual_block(x, channels, train_flag, stride=1):
    filters_in = x.get_shape()[-1]
    with tf.variable_scope('blockA'):
        W1 = weight_variable([3, 3, filters_in, channels])
        B1 = bias_variable([channels])
        output1 = tf.nn.relu(batch_norm(x, train_flag))
        conv_1 = conv2d(output1, W1, stride) + B1
        # print(conv_1.shape)
    with tf.variable_scope('blockB'):
        W2 = weight_variable([3, 3, conv_1.get_shape()[-1], channels])
        B2 = bias_variable([channels])
        output2 = tf.nn.relu(batch_norm(conv_1, train_flag))
        conv_2 = conv2d(output2, W2, stride=1) + B2
    with tf.variable_scope('merge'):
        if stride > 1:
            W3 = weight_variable([1, 1, filters_in, channels])
            B3 = bias_variable([channels])
            projection = conv2d(output1, W3, stride=2) + B3
            block = tf.add(conv_2, projection)
        else:
            block = tf.add(conv_2, x)
    return block


def vgg19(image, train_flag):
    with tf.variable_scope('conv1_1'):
        w_conv1_1 = weight_variable([3, 3, 3, 64])
        b_conv1_1 = bias_variable([64])
        output = tf.nn.relu(batch_norm(conv2d(image, w_conv1_1, 1) + b_conv1_1, train_flag))

    with tf.variable_scope('conv1_2'):
        w_conv1_2 = weight_variable([3, 3, 64, 64])
        b_conv1_2 = bias_variable([64])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv1_2, 1) + b_conv1_2, train_flag))

    with tf.variable_scope('maxpool1'):
        output = max_pool_2x2(output)

    with tf.variable_scope('conv2_1'):
        w_conv2_1 = weight_variable([3, 3, 64, 128])
        b_conv2_1 = bias_variable([128])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv2_1, 1) + b_conv2_1, train_flag))

    with tf.variable_scope('conv2_2'):
        w_conv2_2 = weight_variable([3, 3, 128, 128])
        b_conv2_2 = bias_variable([128])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv2_2, 1) + b_conv2_2, train_flag))

    with tf.variable_scope('maxpool2'):
        output = max_pool_2x2(output)

    with tf.variable_scope('conv3_1'):
        w_conv3_1 = weight_variable([3, 3, 128, 256])
        b_conv3_1 = bias_variable([256])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv3_1, 1) + b_conv3_1, train_flag))

    with tf.variable_scope('conv3_2'):
        w_conv3_2 = weight_variable([3, 3, 256, 256])
        b_conv3_2 = bias_variable([256])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv3_2, 1) + b_conv3_2, train_flag))

    with tf.variable_scope('conv3_3'):
        w_conv3_3 = weight_variable([3, 3, 256, 256])
        b_conv3_3 = bias_variable([256])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv3_3, 1) + b_conv3_3, train_flag))

    with tf.variable_scope('conv3_4'):
        w_conv3_4 = weight_variable([3, 3, 256, 256])
        b_conv3_4 = bias_variable([256])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv3_4, 1) + b_conv3_4, train_flag))

    with tf.variable_scope('maxpool3'):
        output = max_pool_2x2(output)

    with tf.variable_scope('conv4_1'):
        w_conv4_1 = weight_variable([3, 3, 256, 512])
        b_conv4_1 = bias_variable([512])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv4_1, 1) + b_conv4_1, train_flag))

    with tf.variable_scope('conv4_2'):
        w_conv4_2 = weight_variable([3, 3, 512, 512])
        b_conv4_2 = bias_variable([512])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv4_2, 1) + b_conv4_2, train_flag))

    with tf.variable_scope('conv4_3'):
        w_conv4_3 = weight_variable([3, 3, 512, 512])
        b_conv4_3 = bias_variable([512])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv4_3, 1) + b_conv4_3, train_flag))

    with tf.variable_scope('conv4_4'):
        w_conv4_4 = weight_variable([3, 3, 512, 512])
        b_conv4_4 = bias_variable([512])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv4_4, 1) + b_conv4_4, train_flag))

    with tf.variable_scope('maxpool4'):
        output = max_pool_2x2(output)

    with tf.variable_scope('conv5_1'):
        w_conv5_1 = weight_variable([3, 3, 512, 512])
        b_conv5_1 = bias_variable([512])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv5_1, 1) + b_conv5_1, train_flag))

    with tf.variable_scope('conv5_2'):
        w_conv5_2 = weight_variable([3, 3, 512, 512])
        b_conv5_2 = bias_variable([512])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv5_2, 1) + b_conv5_2, train_flag))

    with tf.variable_scope('conv5_3'):
        w_conv5_3 = weight_variable([3, 3, 512, 512])
        b_conv5_3 = bias_variable([512])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv5_3, 1) + b_conv5_3, train_flag))

    with tf.variable_scope('conv5_4'):
        w_conv5_4 = weight_variable([3, 3, 512, 512])
        b_conv5_4 = bias_variable([512])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv5_4, 1) + b_conv5_4, train_flag))

    output = tf.reshape(output, [-1, 2 * 2 * 512])

    with tf.variable_scope('fc1'):
        w_fc1 = weight_variable([2 * 2 * 512, 4096])
        b_fc1 = bias_variable([4096])
        output = tf.nn.relu(batch_norm(tf.matmul(output, w_fc1) + b_fc1, train_flag))

    with tf.variable_scope('fc2'):
        w_fc2 = weight_variable([4096, 4096])
        b_fc2 = bias_variable([4096])
        output = tf.matmul(output, w_fc2) + b_fc2
        intermediate = output
        output = tf.nn.relu(batch_norm(output, train_flag))

    with tf.variable_scope('fc3'):
        w_fc3 = weight_variable([4096, 10])
        b_fc3 = bias_variable([10])
        output = tf.matmul(output, w_fc3) + b_fc3

    return output, intermediate


def vgg19_mod(image, train_flag, keep_prob):
    with tf.variable_scope('conv1_1'):
        w_conv1_1 = weight_variable([3, 3, 3, 64])
        b_conv1_1 = bias_variable([64])
        output = tf.nn.relu(batch_norm(conv2d(image, w_conv1_1, 1) + b_conv1_1, train_flag))

    with tf.variable_scope('conv1_2'):
        w_conv1_2 = weight_variable([3, 3, 64, 64])
        b_conv1_2 = bias_variable([64])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv1_2, 1) + b_conv1_2, train_flag))

    with tf.variable_scope('maxpool1'):
        output = max_pool_2x2(output)

    with tf.variable_scope('conv2_1'):
        w_conv2_1 = weight_variable([3, 3, 64, 128])
        b_conv2_1 = bias_variable([128])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv2_1, 1) + b_conv2_1, train_flag))

    with tf.variable_scope('conv2_2'):
        w_conv2_2 = weight_variable([3, 3, 128, 128])
        b_conv2_2 = bias_variable([128])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv2_2, 1) + b_conv2_2, train_flag))

    with tf.variable_scope('maxpool2'):
        output = max_pool_2x2(output)

    with tf.variable_scope('conv3_1'):
        w_conv3_1 = weight_variable([3, 3, 128, 256])
        b_conv3_1 = bias_variable([256])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv3_1, 1) + b_conv3_1, train_flag))

    with tf.variable_scope('conv3_2'):
        w_conv3_2 = weight_variable([3, 3, 256, 256])
        b_conv3_2 = bias_variable([256])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv3_2, 1) + b_conv3_2, train_flag))

    with tf.variable_scope('conv3_3'):
        w_conv3_3 = weight_variable([3, 3, 256, 256])
        b_conv3_3 = bias_variable([256])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv3_3, 1) + b_conv3_3, train_flag))

    with tf.variable_scope('conv3_4'):
        w_conv3_4 = weight_variable([3, 3, 256, 256])
        b_conv3_4 = bias_variable([256])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv3_4, 1) + b_conv3_4, train_flag))

    with tf.variable_scope('maxpool3'):
        output = max_pool_2x2(output)

    with tf.variable_scope('conv4_1'):
        w_conv4_1 = weight_variable([3, 3, 256, 512])
        b_conv4_1 = bias_variable([512])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv4_1, 1) + b_conv4_1, train_flag))

    with tf.variable_scope('conv4_2'):
        w_conv4_2 = weight_variable([3, 3, 512, 512])
        b_conv4_2 = bias_variable([512])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv4_2, 1) + b_conv4_2, train_flag))

    with tf.variable_scope('conv4_3'):
        w_conv4_3 = weight_variable([3, 3, 512, 512])
        b_conv4_3 = bias_variable([512])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv4_3, 1) + b_conv4_3, train_flag))

    with tf.variable_scope('conv4_4'):
        w_conv4_4 = weight_variable([3, 3, 512, 512])
        b_conv4_4 = bias_variable([512])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv4_4, 1) + b_conv4_4, train_flag))

    with tf.variable_scope('maxpool4'):
        output = max_pool_2x2(output)

    with tf.variable_scope('conv5_1'):
        w_conv5_1 = weight_variable([3, 3, 512, 512])
        b_conv5_1 = bias_variable([512])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv5_1, 1) + b_conv5_1, train_flag))

    with tf.variable_scope('conv5_2'):
        w_conv5_2 = weight_variable([3, 3, 512, 512])
        b_conv5_2 = bias_variable([512])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv5_2, 1) + b_conv5_2, train_flag))

    with tf.variable_scope('conv5_3'):
        w_conv5_3 = weight_variable([3, 3, 512, 512])
        b_conv5_3 = bias_variable([512])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv5_3, 1) + b_conv5_3, train_flag))

    with tf.variable_scope('conv5_4'):
        w_conv5_4 = weight_variable([3, 3, 512, 512])
        b_conv5_4 = bias_variable([512])
        output = tf.nn.relu(batch_norm(conv2d(output, w_conv5_4, 1) + b_conv5_4, train_flag))

    output = tf.reshape(output, [-1, 2 * 2 * 512])

    with tf.variable_scope('fc1'):
        w_fc1 = weight_variable([2 * 2 * 512, 4096])
        b_fc1 = bias_variable([4096])
        output = tf.nn.relu(batch_norm(tf.matmul(output, w_fc1) + b_fc1, train_flag))
    output = tf.nn.dropout(output, keep_prob)

    with tf.variable_scope('fc2'):
        w_fc2 = weight_variable([4096, 1024])
        b_fc2 = bias_variable([1024])
        output = tf.nn.relu(batch_norm(tf.matmul(output, w_fc2) + b_fc2, train_flag))
    output = tf.nn.dropout(output, keep_prob)

    with tf.variable_scope('fc3'):
        w_fc3 = weight_variable([1024, 100])
        b_fc3 = bias_variable([100])
        output = tf.matmul(output, w_fc3) + b_fc3
        intermediate = output
        output = tf.nn.relu(batch_norm(output, train_flag))
    output = tf.nn.dropout(output, keep_prob)

    with tf.variable_scope('fc4'):
        w_fc4 = weight_variable([100, 10])
        b_fc4 = bias_variable([10])
        output = tf.matmul(output, w_fc4) + b_fc4

    return output, intermediate


def resnet(image, train_flag):
    with tf.variable_scope('conv1'):
        w_conv1 = weight_variable([3, 3, 3, 16])
        b_conv1 = bias_variable([16])
        output = conv2d(image, w_conv1) + b_conv1

    for i in range(5):
        with tf.variable_scope('resblock1_' + str(i+1)):
            output = residual_block(output, 16, train_flag)

    with tf.variable_scope('resblock2_1'):
        output = residual_block(output, 32, train_flag, stride=2)
    for i in range(1, 5):
        with tf.variable_scope('resblock2_' + str(i+1)):
            output = residual_block(output, 32, train_flag)

    with tf.variable_scope('resblock3_1'):
        output = residual_block(output, 64, train_flag, stride=2)
    for i in range(1, 5):
        with tf.variable_scope('resblock3_' + str(i+1)):
            output = residual_block(output, 64, train_flag)

    with tf.variable_scope('avg_pool'):
        output = tf.nn.relu(batch_norm(output, train_flag))
        output = global_avg_pool(output, 8, 1)
    output = tf.reshape(output, [-1, 64])
    intermediate = output
    # print(output.shape)

    with tf.variable_scope('fc'):
        w_fc = weight_variable([64, 10])
        b_fc = bias_variable(10)
        output = tf.matmul(output, w_fc) + b_fc

    return output, intermediate
