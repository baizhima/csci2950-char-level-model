from adict import adict
import tensorflow as tf

# From homework 5
def weight_variable(shape):
    return tf.get_variable('w', shape)

def bias_variable(shape):
    return tf.get_variable('b', shape)

def conv2d(input_mat, output_size, kernel_height, kernel_width, scope):
    with tf.variable_scope(scope):
        input_size = input_mat.get_shape()[-1]
        weight_shape = [kernel_height, kernel_width, input_size, output_size]
        w = weight_variable(weight_shape)
        b = bias_variable([output_size])
        return tf.nn.conv2d(input_mat, w, strides=[1, 1, 1, 1], padding='VALID') + b

# just simply linear transform (X*W+B)
def linear_transform(input_mat, output_size, scope):
    with tf.variable_scope(scope):
        input_size = input_mat.get_shape().as_list()[1]
        weight_shape = [output_size, input_size]
        w = tf.get_variable("w", weight_shape)
        b = tf.get_variable("b", [output_size])

        # w = weight_variable([output_size, input_size])
        # b = bias_variable([output_size])
        return tf.matmul(input_mat, tf.transpose(w)) + b


def time_delayed_network(input_mat, scope):
    with tf.variable_scope(scope):
        # from the paper, features
        patch_sizes     = [ 1,   2,   3,   4,   5,   6,   7]
        num_channels    = [50, 100, 150, 200, 200, 200, 200]

        max_word_length = input_mat.get_shape()[1]
        input_mat = tf.expand_dims(input_mat, 1)
        layers = []
        for i in range(len(patch_sizes)):
            patch_size, output_size = patch_sizes[i], num_channels[i]
            new_length = max_word_length - patch_size + 1
            # house keeping conv and pool
            conv = conv2d(input_mat, output_size, 1, patch_size, "patch%d" % patch_size)
            pool_w = [1, 1, new_length, 1]
            h_pool = tf.nn.max_pool(tf.tanh(conv), pool_w, [1, 1, 1, 1], 'VALID')
            layers.append(tf.squeeze(h_pool, [1, 2]))

        return tf.concat(1, layers)


def char_aware_network(char_level_set_size, 
                    word_level_set_size,
                    batch_sz=20,
                    num_unrolls=35,
                    dropout=0.0,
                    max_word_length=65):

    input_mat = tf.placeholder(tf.int32, shape=[batch_sz, num_unrolls, max_word_length], name="input")

    # embedding
    with tf.variable_scope('embed'):
        embedding_size=15
        char_level_shape = [char_level_set_size, embedding_size]
        embedding_char_level = tf.get_variable('embedding_char_level', char_level_shape)
        scatter_shape = [1, embedding_size]
        scatter_char_embed = tf.scatter_update(embedding_char_level, [0], tf.constant(0.0, shape=scatter_shape))
        embedding_shape = [-1, max_word_length, embedding_size]
        embedding_input_mat = tf.reshape(tf.nn.embedding_lookup(embedding_char_level, input_mat), embedding_shape)


    # convolutions
    input_to_network = time_delayed_network(embedding_input_mat, "time_delayed")

    # highway (refer to the paper)
    def highway_transform(input_mat, output_size, num_layers, scope):
        with tf.variable_scope(scope):
            b = -2.0
            for i in range(num_layers):
                # G: non-linearilty, e.g. relu
                non_linearity = tf.nn.relu(linear_transform(input_mat, output_size, scope='hw_g%d' % i))

                # t: transform gate
                transform = tf.sigmoid(linear_transform(input_mat, output_size, scope='hw_t%d' % i) + b)

                # t*g: what non-linearility info to transform
                # (1-t)*input: what info to carry (carry gate)
                output = transform * non_linearity + (1.0 - transform) * input_mat

                input_mat = output

            return output

    input_to_network = highway_transform(input_to_network, input_to_network.get_shape()[-1], num_layers=2, scope='highway')

    # LSTM
    with tf.variable_scope('LSTM'):
        rnn_cell_size = 650
        rnn_layer_size = 2

        cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_cell_size, state_is_tuple=True, forget_bias=0.0)

        # dropout
        if dropout > 0.0:
            keep_prob = 1.0-dropout
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

        # multi layer
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * rnn_layer_size, state_is_tuple=True)

        state_init = cell.zero_state(batch_sz, dtype=tf.float32)

        input_to_network = tf.reshape(input_to_network, [batch_sz, num_unrolls, -1])
        input_to_network2 = [tf.squeeze(x, [1]) for x in tf.split(1, num_unrolls, input_to_network)]

        outputs, state_end = tf.nn.rnn(cell, input_to_network2, initial_state=state_init, dtype=tf.float32)

        logits = []
        with tf.variable_scope('word_embed_linear') as scope:
            for idx, output in enumerate(outputs):
                if idx > 0:
                    scope.reuse_variables()
                logits.append(linear_transform(output, word_level_set_size, "scope_linear_transform"))

    return adict(
        input = input_mat,
        scatter_char_embed=scatter_char_embed,
        state_init=state_init,
        state_end=state_end,
        input_to_network=input_to_network,
        rnn_outputs=outputs,
        embedding_input_mat=embedding_input_mat,
        logits = logits
    )


def cost(logits, batch_sz, num_unrolls):
    with tf.variable_scope('cost'):
        shape = [batch_sz, num_unrolls]
        target_var = tf.placeholder(tf.int64, shape, name='target_var')
        target_list = [tf.squeeze(d, [1]) for d in tf.split(1, num_unrolls, target_var)]
        loss_penalty = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, target_list), name='loss_penalty')
    return adict(
        target_var=target_var,
        loss=loss_penalty
    )



def training_graph(loss, learning_rate):

    with tf.variable_scope('stochastic'):
        glob_step = tf.Variable(0, name='glob_step', trainable=False)
        learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate')
        trainable_vars = tf.trainable_variables()
        grads, glob_norm = tf.clip_by_global_norm(tf.gradients(loss, trainable_vars), 5.0)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = optimizer.apply_gradients(zip(grads, trainable_vars), global_step=glob_step)

    return adict(
        learning_rate=learning_rate,
        train_step=train_step,
        glob_norm=glob_norm,
        glob_step=glob_step)
