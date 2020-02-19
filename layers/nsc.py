import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xavier
from layers.hop import hop
from layers.lstm import lstm
from layers.nsc_sentence_layer import nsc_sentence_layer
from layers.nsc_document_layer import nsc_document_layer


def nsc(x,
        max_sen_len,
        max_doc_len,
        sen_len,
        doc_len,
        identities,
        hidden_size,
        emb_dim,
        augment_document=False,
        sen_hop_cnt=1,
        doc_hop_cnt=1):
    x = tf.reshape(x, [-1, max_sen_len, emb_dim])
    sen_len = tf.reshape(sen_len, [-1])

    #  with tf.variable_scope('sentence_layer'):
    #      # lstm_outputs, _state = lstm(x, sen_len, hidden_size, 'lstm')
    #      # lstm_outputs = tf.reshape(lstm_outputs, [-1, max_sen_len, hidden_size])
    #      lstm_bkg, _state = lstm(x, sen_len, hidden_size, 'lstm_bkg')
    #      lstm_bkg = tf.reshape(lstm_bkg, [-1, max_sen_len, hidden_size])
    #      lstm_outputs = lstm_bkg
    #
    #      sen_bkg = [
    #          tf.reshape(
    #              tf.tile(bkg[:, None, :], (1, max_doc_len, 1)), (-1, emb_dim))
    #          for bkg in identities
    #      ]
    #      for ihop in range(sen_hop_cnt):
    #          last = ihop == sen_hop_cnt - 1
    #          sen_bkg[0] = hop('hop', last, lstm_outputs, lstm_bkg, sen_bkg[0],
    #                           sen_bkg[1:], sen_len, max_sen_len, '')
    #  outputs = tf.reshape(sen_bkg[0], [-1, max_doc_len, hidden_size])
    sen_emb = nsc_sentence_layer(x, max_sen_len, max_doc_len, sen_len,
                                 identities, hidden_size, emb_dim, sen_hop_cnt)
    if augment_document:
        sen_emb = tf.concat([identities[0][:, None, :], sen_emb], axis=1)
        max_doc_len += 1
        doc_len = doc_len + 1

    #  with tf.variable_scope('document_layer'):
    #      # lstm_outputs, _state = lstm(outputs, doc_len, hidden_size, 'lstm')
    #      lstm_bkg, _state = lstm(outputs, doc_len, hidden_size, 'lstm_bkg')
    #      lstm_outputs = lstm_bkg
    #
    #      doc_bkg = [i for i in identities]
    #      for ihop in range(doc_hop_cnt):
    #          last = ihop == doc_hop_cnt - 1
    #          doc_bkg[0] = hop('hop', last, lstm_outputs, lstm_bkg, doc_bkg[0],
    #                           doc_bkg[1:], doc_len, max_doc_len, '')
    #  outputs = doc_bkg[0]
    doc_emb = nsc_document_layer(sen_emb, max_doc_len, doc_len, identities,
                                 hidden_size, doc_hop_cnt)

    return doc_emb
