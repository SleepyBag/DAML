from functools import partial
from math import sqrt
import tensorflow as tf
from tensorflow import constant as const
from tensorflow.nn import embedding_lookup as lookup
from layers.nsc_sentence_layer import nsc_sentence_layer
from layers.nsc_document_layer import nsc_document_layer


def var(name, shape, initializer):
    return tf.get_variable(name, shape=shape, initializer=initializer)


class MHNSC(object):
    def __init__(self, args, wrd_emb):
        self.max_doc_len = args['max_doc_len']
        self.max_sen_len = args['max_sen_len']
        self.cls_cnt = args['cls_cnt']
        self.embedding = args['embedding']
        self.emb_dim = args['emb_dim']
        self.hidden_size = args['hidden_size']
        self.usr_cnt = args['usr_cnt']
        self.prd_cnt = args['prd_cnt']
        self.doc_cnt = args['doc_cnt']
        self.sen_hop_cnt = args['sen_hop_cnt']
        self.doc_hop_cnt = args['doc_hop_cnt']
        self.l2_rate = args['l2_rate']
        self.convert_flag = ''
        self.debug = args['debug']
        self.lambda1 = args['lambda1']
        self.lambda2 = args['lambda2']
        self.lambda3 = args['lambda3']
        self.embedding_lr = args['embedding_lr']

        self.best_dev_acc = .0
        self.best_test_acc = .0
        self.best_test_rmse = .0

        # initializers for parameters
        self.w_init = tf.contrib.layers.xavier_initializer()
        self.b_init = tf.initializers.zeros()
        self.e_init = tf.contrib.layers.xavier_initializer()

        self.wrd_emb = wrd_emb
        self.usr_emb = var('usr_emb', [self.usr_cnt, self.emb_dim],
                           self.e_init)
        self.prd_emb = var('prd_emb', [self.prd_cnt, self.emb_dim],
                           self.e_init)
        self.embeddings = [self.wrd_emb, self.usr_emb, self.prd_emb]

    def build(self, input_map):
        transform = partial(
            tf.layers.dense,
            use_bias=False,
            kernel_initializer=self.w_init,
            bias_initializer=self.b_init)
        dense = partial(
            tf.layers.dense,
            kernel_initializer=self.w_init,
            bias_initializer=self.b_init)
        lstm_cell = partial(
            tf.nn.rnn_cell.LSTMCell,
            self.hidden_size // 2,
            forget_bias=0.,
            initializer=self.w_init)

        def pad_context(context, input_x):
            """ padding content with context embedding """
            tiled_context = transform(context, self.emb_dim)
            tiled_context = tf.tile(tiled_context[:, None, None, :],
                                    [1, self.max_doc_len, 1, 1])
            input_x = tf.reshape(
                input_x,
                [-1, self.max_doc_len, self.max_sen_len, self.emb_dim])
            input_x = tf.concat([tiled_context, input_x], axis=2)
            input_x = tf.reshape(input_x,
                                 [-1, self.max_sen_len + 1, self.emb_dim])
            return input_x

        # get the inputs
        with tf.variable_scope('inputs'):
            usrid, prdid, input_x, input_y, sen_len, doc_len, docid = \
                (input_map['usr'], input_map['prd'],
                 input_map['content'], input_map['rating'],
                 input_map['sen_len'], input_map['doc_len'],
                 input_map['docid'])

            usr = lookup(self.usr_emb, usrid)
            prd = lookup(self.prd_emb, prdid)
            input_x = lookup(self.wrd_emb, input_x)

        nscua_input_x = pad_context(usr, input_x)
        nscpa_input_x = pad_context(prd, input_x)
        nscua_input_x = input_x
        nscpa_input_x = input_x

        #  sen_len = tf.where(
        #      tf.equal(sen_len, 0), tf.zeros_like(sen_len), sen_len + 1)
        #  self.max_sen_len += 1

        # build the process of model
        sen_embs, doc_embs = [], []
        sen_cell_fw = lstm_cell()
        sen_cell_bw = lstm_cell()
        for scope, identities, input_x, attention_type in zip(
                ['user_block', 'product_block'], [[usr], [prd]],
                [nscua_input_x, nscpa_input_x], ['additive', 'additive']):
            with tf.variable_scope(scope):
                sen_emb = nsc_sentence_layer(
                    input_x,
                    self.max_sen_len,
                    self.max_doc_len,
                    sen_len,
                    identities,
                    self.hidden_size,
                    self.emb_dim,
                    self.sen_hop_cnt,
                    bidirectional_lstm=True,
                    lstm_cells=[sen_cell_fw, sen_cell_bw],
                    auged=False,
                    attention_type=attention_type)
                sen_embs.append(sen_emb)

        sen_embs = tf.concat(sen_embs, axis=-1)

        # padding doc with user and product embeddings
        doc_aug_usr = transform(usr, 2 * self.hidden_size)
        nscua_sen_embs = tf.concat([doc_aug_usr[:, None, :], sen_embs], axis=1)
        doc_aug_prd = transform(prd, 2 * self.hidden_size)
        nscpa_sen_embs = tf.concat([doc_aug_prd[:, None, :], sen_embs], axis=1)
        #  none_sen_embs = tf.pad(sen_embs, [[0, 0], [1, 0], [0, 0]])
        nscua_sen_embs = sen_embs
        nscpa_sen_embs = sen_embs
        #  self.max_doc_len += 1
        #  doc_len = doc_len + 1

        doc_cell_fw = lstm_cell()
        doc_cell_bw = lstm_cell()
        for scope, identities, input_x, attention_type in zip(
                ['user_block', 'product_block'], [[usr], [prd]],
                [nscua_sen_embs, nscpa_sen_embs], ['additive', 'additive']):
            with tf.variable_scope(scope):
                doc_emb = nsc_document_layer(
                    input_x,
                    self.max_doc_len,
                    doc_len,
                    identities,
                    self.hidden_size,
                    self.doc_hop_cnt,
                    bidirectional_lstm=True,
                    lstm_cells=[doc_cell_fw, doc_cell_bw],
                    auged=False,
                    attention_type=attention_type)
                doc_embs.append(doc_emb)

        with tf.variable_scope('result'):
            doc_emb = tf.concat(doc_embs, axis=1, name='dhuapa_output')
            logit = dense(doc_emb, self.cls_cnt)
        return logit
