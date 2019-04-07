import tensorflow as tf
import numpy as np
import math
import tensorflow.contrib.slim as slim

def __dropout__(x, keep_prob=1.0):
    return tf.contrib.layers.dropout(x, keep_prob=keep_prob)

def __pooling__(x):
    return tf.reduce_max(x, axis=-2)

def __piecewise_pooling__(x, mask):
    mask_embedding = tf.constant([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    mask = tf.nn.embedding_lookup(mask_embedding, mask)
    hidden_size = x.shape[-1]
    x = tf.reduce_max(tf.expand_dims(mask * 100, 2) + tf.expand_dims(x, 3), axis=1) - 100
    return tf.reshape(x, [-1, hidden_size * 3])

def __cnn_cell__(x, hidden_size=230, kernel_size=3, stride_size=1):
    x = tf.layers.conv1d(inputs=x, 
                         filters=hidden_size, 
                         kernel_size=kernel_size, 
                         strides=stride_size, 
                         padding='same', 
                         kernel_initializer=tf.contrib.layers.xavier_initializer())
    return x

def cnn(x, hidden_size=230, kernel_size=3, stride_size=1, activation=tf.nn.relu, var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "cnn", reuse=tf.AUTO_REUSE):
        max_length = x.shape[1]
        x = __cnn_cell__(x, hidden_size, kernel_size, stride_size)
        x = __pooling__(x)
        x = activation(x)
        x = __dropout__(x, keep_prob)
        return x

def pcnn(x, mask, hidden_size=230, kernel_size=3, stride_size=1, activation=tf.nn.relu, var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "pcnn", reuse=tf.AUTO_REUSE):
        max_length = x.shape[1]
        x = __cnn_cell__(x, hidden_size, kernel_size, stride_size)
        x = __piecewise_pooling__(x, mask)
        x = activation(x)
        x = __dropout__(x, keep_prob)
        return x

def __rnn_cell__(hidden_size, cell_name='lstm'):
    if isinstance(cell_name, list) or isinstance(cell_name, tuple):
        if len(cell_name) == 1:
            return __rnn_cell__(hidden_size, cell_name[0])
        cells = [self.__rnn_cell__(hidden_size, c) for c in cell_name]
        return tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
    if cell_name.lower() == 'lstm':
        return tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    elif cell_name.lower() == 'gru':
        return tf.contrib.rnn.GRUCell(hidden_size)
    raise NotImplementedError

def rnn(x, length, hidden_size=230, cell_name='lstm', var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "rnn", reuse=tf.AUTO_REUSE):
        x = __dropout__(x, keep_prob)
        cell = __rnn_cell__(hidden_size, cell_name)
        _, states = tf.nn.dynamic_rnn(cell, x, sequence_length=length, dtype=tf.float32, scope='dynamic-rnn')
        if isinstance(states, tuple):
            states = states[0]
        return states

def birnn(x, length, hidden_size=230, cell_name='lstm', var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "birnn", reuse=tf.AUTO_REUSE):
        x = __dropout__(x, keep_prob)
        fw_cell = __rnn_cell__(hidden_size, cell_name)
        bw_cell = __rnn_cell__(hidden_size, cell_name)
        _, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, sequence_length=length, dtype=tf.float32, scope='dynamic-bi-rnn')
        fw_states, bw_states = states
        if isinstance(fw_states, tuple):
            fw_states = fw_states[0]
            bw_states = bw_states[0]
        return tf.concat([fw_states, bw_states], axis=1)


def capsnn(x, hidden_size=100, kernel_size=(3, 60), stride_size=(1, 1), var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "capsnn", reuse=tf.AUTO_REUSE):
        x = tf.expand_dims(x, -1)

        with tf.variable_scope('PrimaryCaps_layer'):
            output = slim.conv2d(x, num_outputs=hidden_size * 4, kernel_size=kernel_size, stride=stride_size, padding='VALID',
                                 activation_fn=None)
            output = tf.reshape(output, [-1, output.shape[1] * output.shape[2] * output.shape[3] / 4, 1, 4])

            output = squash(output)  # [batch_size,1152,1,4]
        with tf.variable_scope('DigitCaps_layer'):
            u_hats = []
            input_groups = tf.split(axis=1, num_or_size_splits=output.shape[1], value=output)
            for i in range(output.shape[1]):
                u_hat = slim.conv2d(input_groups[i], num_outputs=16 * 20, kernel_size=[1, 1], stride=1, padding='VALID',
                                    scope='DigitCaps_layer_w_' + str(i), activation_fn=None)
                u_hat = tf.reshape(u_hat, [-1, 1, 20, 16])
                u_hats.append(u_hat)

            output = tf.concat(u_hats, axis=1)
            assert output.get_shape() == [-1, 1152, 20, 16]

            b_ijs = tf.constant(np.zeros([output.shape[1], 20], dtype=np.float32))
            v_js = []
            for r_iter in range(3):
                with tf.variable_scope('iter_' + str(r_iter)):
                    c_ijs = tf.nn.softmax(b_ijs, dim=1)

                    c_ij_groups = tf.split(axis=1, num_or_size_splits=20, value=c_ijs)
                    b_ij_groups = tf.split(axis=1, num_or_size_splits=20, value=b_ijs)
                    input_groups = tf.split(axis=2, num_or_size_splits=20, value=output)

                    for i in range(20):
                        c_ij = tf.reshape(tf.tile(c_ij_groups[i], [1, 16]), [output.shape[1], 1, 16, 1])
                        s_j = tf.nn.depthwise_conv2d(input_groups[i], c_ij, strides=[1, 1, 1, 1], padding='VALID')

                        s_j = tf.reshape(s_j, [-1, 16])
                        s_j_norm_square = tf.reduce_mean(tf.square(s_j), axis=1, keep_dims=True)
                        v_j = s_j_norm_square * s_j / ((1 + s_j_norm_square) * tf.sqrt(s_j_norm_square + 1e-9))

                        b_ij_groups[i] = b_ij_groups[i] + tf.reduce_sum(
                            tf.matmul(tf.reshape(input_groups[i], [-1, output.shape[1], 16]),
                                      tf.reshape(v_j, [-1, 16, 1])), axis=0)

                        if r_iter == 2:
                            v_js.append(tf.reshape(v_j, [-1, 1, 16]))

                    b_ijs = tf.concat(b_ij_groups, axis=1)

            output = tf.concat(v_js, axis=1)

        x = tf.reshape(output, [-1, output.shape[1]*output.shape[2]])
        x = __dropout__(x, keep_prob)
        return x


epsilon = 1e-9


def squash(vector):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)
