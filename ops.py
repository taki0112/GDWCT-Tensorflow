import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np
from utils import pytorch_xavier_weight_factor, pytorch_kaiming_weight_factor

factor, mode, uniform = pytorch_kaiming_weight_factor(a=0.0, uniform=False)
weight_init = tf_contrib.layers.variance_scaling_initializer(factor=factor, mode=mode, uniform=uniform)

# weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)

weight_regularizer = tf_contrib.layers.l2_regularizer(scale=0.0001)
weight_regularizer_fully = tf_contrib.layers.l2_regularizer(scale=0.0001)

##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):

        if scope.__contains__("discriminator"):
            weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        else:
            weight_init = tf_contrib.layers.variance_scaling_initializer(factor=factor, mode=mode, uniform=uniform)

        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)

        return x


def partial_conv(x, channels, kernel=3, stride=2, use_bias=True, padding='SAME', sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if padding.lower() == 'SAME'.lower():
            with tf.variable_scope('mask'):
                _, h, w, _ = x.get_shape().as_list()

                slide_window = kernel * kernel
                mask = tf.ones(shape=[1, h, w, 1])

                update_mask = tf.layers.conv2d(mask, filters=1,
                                               kernel_size=kernel, kernel_initializer=tf.constant_initializer(1.0),
                                               strides=stride, padding=padding, use_bias=False, trainable=False)

                mask_ratio = slide_window / (update_mask + 1e-8)
                update_mask = tf.clip_by_value(update_mask, 0.0, 1.0)
                mask_ratio = mask_ratio * update_mask

            with tf.variable_scope('x'):
                if sn:
                    w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels],
                                        initializer=weight_init, regularizer=weight_regularizer)
                    x = tf.nn.conv2d(input=x, filter=spectral_norm(w), strides=[1, stride, stride, 1], padding=padding)
                else:
                    x = tf.layers.conv2d(x, filters=channels,
                                         kernel_size=kernel, kernel_initializer=weight_init,
                                         kernel_regularizer=weight_regularizer,
                                         strides=stride, padding=padding, use_bias=False)
                x = x * mask_ratio

                if use_bias:
                    bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))

                    x = tf.nn.bias_add(x, bias)
                    x = x * update_mask
        else:
            if sn:
                w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels],
                                    initializer=weight_init, regularizer=weight_regularizer)
                x = tf.nn.conv2d(input=x, filter=spectral_norm(w), strides=[1, stride, stride, 1], padding=padding)
                if use_bias:
                    bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))

                    x = tf.nn.bias_add(x, bias)
            else:
                x = tf.layers.conv2d(x, filters=channels,
                                     kernel_size=kernel, kernel_initializer=weight_init,
                                     kernel_regularizer=weight_regularizer,
                                     strides=stride, padding=padding, use_bias=use_bias)

        return x

def fully_connected(x, units, use_bias=True, sn=False, scope='linear'):
    with tf.variable_scope(scope):
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if sn:
            w = tf.get_variable("kernel", [channels, units], tf.float32,
                                initializer=weight_init, regularizer=weight_regularizer_fully)
            if use_bias:
                bias = tf.get_variable("bias", [units],
                                       initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else:
                x = tf.matmul(x, spectral_norm(w))

        else:
            x = tf.layers.dense(x, units=units, kernel_initializer=weight_init,
                                kernel_regularizer=weight_regularizer_fully,
                                use_bias=use_bias)

        return x

def flatten(x) :
    return tf.layers.flatten(x)

def GDWCT_block(content, style, style_mu, group_num) :

    content = deep_whitening_transform(content) # [bs, h, w, ch]
    U, style = deep_coloring_transform(style, group_num) # [bs, 1, ch], [bs, ch, ch]

    bs, h, w, ch = content.get_shape().as_list()
    content = tf.reshape(content, shape=[bs, h*w, ch])

    x = tf.matmul(content, style) + style_mu

    x = tf.reshape(x, shape=[bs, h, w, ch])

    return x, U

def deep_whitening_transform(c) :

    mu, _ = tf.nn.moments(c, axes=[1, 2], keep_dims=True)
    x = c - mu

    return x

def deep_coloring_transform(s, group_num) :
    # [batch_size, 1, channel] : S : MLP^CT(s)
    bs, _, ch = s.get_shape().as_list()

    # make U
    l2_norm = tf.norm(s, axis=-1, keepdims=True)
    U = s / l2_norm

    # make D
    eye = tf.eye(num_rows=ch, num_columns=ch, batch_shape=[bs]) # [batch_size, channel, channel]
    D = l2_norm * eye

    U_block_list = []
    split_num = ch // group_num

    for i in range(group_num) :
        U_ = U[:, :, i * split_num: (i + 1) * split_num]
        D_ = D[:, i * split_num: (i + 1) * split_num, i * split_num: (i + 1) * split_num]

        block_matrix = U_ * D_ * tf.transpose(U_, perm=[0, 2, 1])
        operator_matrix = tf.linalg.LinearOperatorFullMatrix(block_matrix)
        U_block_list.append(operator_matrix)

    U_block_diag_matrix = tf.linalg.LinearOperatorBlockDiag(U_block_list).to_dense()

    return U, U_block_diag_matrix

def group_wise_regularization(c_whitening, U_list, group_num) :
    """ Regularization """

    """ whitening regularization """
    bs, h, w, ch = c_whitening.get_shape().as_list()
    c_whitening = tf.reshape(c_whitening, shape=[bs, h * w, ch])
    c_whitening = tf.matmul(tf.transpose(c_whitening, perm=[0, 2, 1]), c_whitening)  # covariance of x [bs, ch, ch]
    bs, ch, _ = c_whitening.get_shape().as_list() # ch1 = ch2

    index_matrix = make_index_matrix(bs, ch, ch, group_num)

    group_convariance_x = tf.where(tf.equal(index_matrix, 1.0), c_whitening, tf.zeros_like(c_whitening))
    group_convariance_x = tf.linalg.set_diag(group_convariance_x, tf.ones([bs, ch]))

    whitening_regularization_loss = L1_loss(c_whitening, group_convariance_x)

    """ coloring regularization """
    split_num = ch // group_num

    coloring_regularization_list = []
    coloring_regularization_loss_list = []

    for U in U_list :
        # [bs, 1, ch]
        for i in range(group_num):
            U_ = U[:, :, i * split_num: (i + 1) * split_num]

            U_TU = tf.matmul(tf.transpose(U_, perm=[0, 2, 1]), U_) # [bs, ch // group_num, ch // group_num]
            coloring_regularization_list.append(L1_loss(U_TU, tf.eye(num_rows=ch // group_num, num_columns=ch // group_num, batch_shape=[bs])))

        coloring_regularization_loss_list.append(tf.reduce_mean(coloring_regularization_list))

    coloring_regularization_loss = tf.reduce_mean(coloring_regularization_loss_list)

    return whitening_regularization_loss, coloring_regularization_loss



def make_index_matrix(bs, ch1, ch2, group_num) :
    index_matrix = np.abs(np.kron(np.eye(ch1 // group_num, ch2 // group_num), np.eye(group_num, group_num) - 1))
    index_matrix[index_matrix == 0] = -1
    index_matrix[index_matrix == 1] = 0
    index_matrix[index_matrix == -1] = 1
    index_matrix = np.tile(index_matrix, [bs, 1, 1])

    return index_matrix

##################################################################################
# Residual-block
##################################################################################

def resblock(x_init, channels, use_bias=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = instance_norm(x)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = instance_norm(x)

        return x + x_init

def no_norm_resblock(x_init, channels, use_bias=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)

        return x + x_init

def group_resblock(x_init, channels, groups, use_bias=True, sn=False, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = group_norm(x, groups)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, sn=sn)
            x = group_norm(x, groups)

        return x + x_init

##################################################################################
# Sampling
##################################################################################

def down_sample(x) :
    return tf.layers.average_pooling2d(x, pool_size=3, strides=2, padding='SAME')

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

def up_sample_nearest(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap

##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.01):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

##################################################################################
# Normalization function
##################################################################################


def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)

def layer_norm(x, scope='layer_norm') :
    return tf_contrib.layers.layer_norm(x,
                                        center=True, scale=True,
                                        scope=scope)

def group_norm(x, groups=8, scope='group_norm'):
    return tf.contrib.layers.group_norm(x, groups=groups, epsilon=1e-05,
                                        center=True, scale=True,
                                        scope=scope)

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

##################################################################################
# Loss function
##################################################################################

def discriminator_loss(gan_type, real, fake):
    n_scale = len(real)
    loss = []

    real_loss = 0
    fake_loss = 0

    for i in range(n_scale) :
        if gan_type == 'lsgan' :
            real_loss = tf.reduce_mean(tf.squared_difference(real[i], 1.0))
            fake_loss = tf.reduce_mean(tf.square(fake[i]))

        if gan_type == 'gan' :
            real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real[i]), logits=real[i]))
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake[i]), logits=fake[i]))

        if gan_type == 'hinge':
            real_loss = -tf.reduce_mean(tf.minimum(real[i][-1] - 1, 0.0))
            fake_loss = -tf.reduce_mean(tf.minimum(-fake[i][-1] - 1, 0.0))

        loss.append(real_loss + fake_loss)

    return tf.reduce_sum(loss)


def generator_loss(gan_type, fake):
    n_scale = len(fake)
    loss = []

    fake_loss = 0

    for i in range(n_scale) :
        if gan_type == 'lsgan' :
            fake_loss = tf.reduce_mean(tf.squared_difference(fake[i], 1.0))

        if gan_type == 'gan' :
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake[i]), logits=fake[i]))

        if gan_type == 'hinge':
            # fake_loss = -tf.reduce_mean(relu(fake[i][-1]))
            fake_loss = -tf.reduce_mean(fake[i][-1])

        loss.append(fake_loss)


    return tf.reduce_sum(loss)


def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss

def regularization_loss(scope_name) :
    """
    If you want to use "Regularization"
    g_loss += regularization_loss('generator')
    d_loss += regularization_loss('discriminator')
    """
    collection_regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss = []
    for item in collection_regularization :
        if scope_name in item.name :
            loss.append(item)

    return tf.reduce_sum(loss)

def z_sample(mean, logvar):
    eps = tf.random_normal(tf.shape(mean), mean=0.0, stddev=1.0, dtype=tf.float32)

    return mean + tf.exp(logvar * 0.5) * eps


def kl_loss(mean, logvar):
    # shape : [batch_size, channel]
    loss = 0.5 * tf.reduce_sum(tf.square(mean) + tf.exp(logvar) - 1 - logvar, axis=-1)
    loss = tf.reduce_mean(loss)

    return loss