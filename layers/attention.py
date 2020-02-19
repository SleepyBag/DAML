from collections import Iterable
import tensorflow as tf


def attention(h,
              bkg,
              doc_len,
              real_max_len,
              biases_initializer=tf.initializers.zeros(),
              weights_initializer=tf.contrib.layers.xavier_initializer(),
              auged=False):
    if bkg is None:
        bkg = []
    max_len = h.shape[1]
    hidden_size = h.shape[2]

    # create variables
    wh = tf.get_variable(
        'wh', [hidden_size, hidden_size], initializer=weights_initializer)
    b = tf.get_variable('b', [hidden_size], initializer=biases_initializer)
    v = tf.get_variable('v', [hidden_size, 1], initializer=biases_initializer)

    h = tf.reshape(h, [-1, hidden_size])
    h = h @ wh
    e = tf.reshape(h + b, [-1, max_len, hidden_size])

    # add influences of background
    for n, bkgi in enumerate(bkg if isinstance(bkg, Iterable) else [bkg]):
        wbkg = tf.get_variable(
            'wbkg' + str(n) + str(bkgi.shape[1]), [bkgi.shape[1], hidden_size],
            initializer=weights_initializer)
        with tf.variable_scope('attention' + str(n)):
            e = e + (bkgi @ wbkg)[:, None, :]

    e = tf.tanh(e)
    e = tf.reshape(e, [-1, hidden_size])
    e = tf.reshape(e @ v, [-1, real_max_len])
    e = tf.nn.softmax(e, name='attention_with_null_word')
    if auged:
        mask = tf.sequence_mask(doc_len, real_max_len - 1, dtype=tf.float32)
        mask = tf.pad(mask, [[0, 0], [1, 0]])
    else:
        mask = tf.sequence_mask(doc_len, real_max_len, dtype=tf.float32)
    e = (e * mask)[:, None, :]
    _sum = tf.reduce_sum(e, reduction_indices=2, keepdims=True) + 1e-9
    e = tf.div(e, _sum, name='attention_without_null_word')
    # e = tf.reshape(e, [-1, max_doc_len], name='attention_without_null_word')

    return e
