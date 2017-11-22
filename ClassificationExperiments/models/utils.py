import tensorflow as tf
from magenta.models.nsynth.wavenet.h512_bo16 import Config


def build_wavenet(batch_size=1, sample_length=64000):
    config = Config()

    x = tf.placeholder(tf.float32, shape=[batch_size, sample_length])
    graph = config.build({"wav": x}, is_training=False)
    graph.update({"X": x})
    return graph


def multi_layer_perceptron(input_tensor, neurons_per_layer, activations=tf.nn.relu, scope='mlp', reuse=None):
    with tf.variable_scope(scope, reuse=reuse) as net_scope:
        x = input_tensor

        # If the input tensor is not 2D we need to split and stack it so that it is
        len_x_shape = len(x.shape)
        while len_x_shape > 2:
            # Unstack along the final axis
            x = tf.unstack(x, axis=len_x_shape - 1)

            # Concatenate along the new final axis
            x = tf.concat(x, axis=len_x_shape - 2)

            # Recompute shape
            len_x_shape = len(x.shape)

        name_count = 0
        if type(activations) is list:
            for n, activation in zip(neurons_per_layer, activations):
                x = tf.layers.dense(x, n, activation, name=str(name_count), reuse=reuse)
                name_count += 1
        else:
            for n in neurons_per_layer[:-1]:
                x = tf.layers.dense(x, n, activations, name=str(name_count), reuse=reuse)
                name_count += 1
            x = tf.layers.dense(x, neurons_per_layer[-1], name=str(name_count), reuse=reuse, use_bias=False)

        return x


def classification_net(sample_embedding, num_classes, lstm_size=64):
    # Lstm for time dim
    rnn_cell = tf.nn.rnn_cell.LSTMCell(lstm_size)

    rnn_output, rnn_state = tf.nn.dynamic_rnn(rnn_cell, sample_embedding, dtype=tf.float32)

    logits = multi_layer_perceptron(rnn_state[-1], [lstm_size, 2], activations=[tf.nn.relu, None]) # TODO - make neurons_per_layer a parameter

    preds = tf.nn.softmax(logits)

    return logits, preds