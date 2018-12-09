from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


tf.set_random_seed(1)

# Hyper Parameters
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
weight_decay = 1e-4
learning_rate = 1e-3
num_classes = 10


def rnn(only_logits=False):
    np.random.seed(1)
    # tensorflow placeholders
    images_ph = tf.placeholder(tf.float32, [None, TIME_STEP * INPUT_SIZE])       # shape(batch, 784)
    labels_ph = tf.placeholder(tf.int32, [None, num_classes])                             # input y
    is_training_ph = tf.placeholder(tf.bool, shape=())
    
    global_step = tf.train.get_or_create_global_step()
    image = tf.reshape(images_ph, [-1, TIME_STEP, INPUT_SIZE])       # (batch, height, width, channel)
    # RNN
    rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=64)
    outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
        rnn_cell,                   # cell you have chosen
        image,                      # input
        initial_state=None,         # the initial hidden state
        dtype=tf.float32,           # must given if set initial_state = None
        time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
    )
    logits = tf.layers.dense(outputs[:, -1, :], num_classes)              # output based on the last output step

    if only_logits:
        return {'logits': logits,
                   'images': images_ph,
                   'labels': labels_ph,
                   'is_training': is_training_ph,
                   'global_step': global_step}
    
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels_ph, logits=logits)           # compute cost
    loss += weight_decay * tf.losses.get_regularization_loss()
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(labels_ph, axis=1)), tf.float32))
#     acc = tf.metrics.accuracy(labels=tf.argmax(labels_ph, axis=1), predictions=tf.argmax(logits, axis=1),)[1]
    
    train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss, global_step=global_step)

    return {'train_op': train_op,
               'logits': logits,
               'loss': loss,
               'acc': acc,
               'images': images_ph,
               'labels': labels_ph,
               'is_training': is_training_ph,
               'global_step': global_step}
