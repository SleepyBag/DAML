from collections import Iterable
import tensorflow as tf
from layers.attention import attention as additive_attention
from layers.dot_attention import dot_attention


def hop(scope,
        sentence,
        sentence_bkg,
        bkg_iter,
        bkg_fix,
        doc_len,
        real_max_len,
        convert_flag,
        biases_initializer=tf.initializers.zeros(),
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        auged=False,
        attention_type='additive',
        extra=''):
    """
    return a new embedding of background information.
    attention_type is one of 'additive' and 'dot', indicating the calculation of attention machenism
    """

    if bkg_fix is None:
        bkg_fix = []
    if not isinstance(bkg_fix, list):
        bkg_fix = [bkg_fix]
    bkg_fix = list(bkg_fix)
    hidden_size = sentence_bkg.shape[2]

    if attention_type == 'additive':
        attention = additive_attention
    elif attention_type == 'dot':
        attention = dot_attention
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if bkg_iter is not None:
            alphas = attention(
                sentence_bkg, [bkg_iter] + bkg_fix,
                doc_len,
                real_max_len,
                biases_initializer=biases_initializer,
                weights_initializer=weights_initializer,
                auged=auged)
        else:
            alphas = attention(
                sentence_bkg,
                bkg_fix,
                doc_len,
                real_max_len,
                biases_initializer=biases_initializer,
                weights_initializer=weights_initializer,
                auged=auged)
        new_bkg = alphas @ sentence
        new_bkg = tf.reshape(new_bkg, [-1, hidden_size])
        if extra == 'jci':
            extra_alpha = (bkg_fix[0][:, None, :] @ tf.transpose(sentence, [0, 2, 1])) * alphas
            extra_component = extra_alpha @ sentence
            extra_component = tf.reshape(extra_component, [-1, hidden_size])
        if extra != '':
            new_bkg = tf.concat([new_bkg, extra_component], axis=1)
        if 'o' in convert_flag:
            new_bkg = bkg_iter + new_bkg
    return new_bkg
