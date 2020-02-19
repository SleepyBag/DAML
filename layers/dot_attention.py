from collections import Iterable
import tensorflow as tf


def dot_attention(h,
                  bkg,
                  doc_len,
                  real_max_len,
                  biases_initializer=tf.initializers.zeros(),
                  weights_initializer=tf.contrib.layers.xavier_initializer(),
                  auged=False):
    if isinstance(bkg, Iterable):
        bkg = bkg[0]
    max_len = h.shape[1]
    hidden_size = h.shape[2]

    # create variables
    w = tf.get_variable('w', [hidden_size, bkg.shape[1]], initializer=weights_initializer)

    h = tf.reshape(h, [-1, hidden_size])
    h = h @ w
    e = tf.reshape(h, [-1, max_len, bkg.shape[1]])
    e = e * bkg[:, None, :]
    e = tf.reduce_sum(e, axis=-1)

    e = tf.reshape(e, [-1, real_max_len])
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
