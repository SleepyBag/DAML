import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xavier
from tensorflow.nn import embedding_lookup as lookup
from layers.hop import hop


def var(name, shape, initializer):
    return tf.get_variable(name, shape=shape, initializer=initializer)


class HUAPA(object):
    def __init__(self, args, wrd_emb):
        self.max_doc_len = args['max_doc_len']
        self.max_sen_len = args['max_sen_len']
        self.cls_cnt = args['cls_cnt']
        self.embedding = args['embedding']
        self.emb_dim = args['emb_dim']
        self.hidden_size = self.emb_dim
        self.usr_cnt = args['usr_cnt']
        self.prd_cnt = args['prd_cnt']
        self.l2_rate = args['l2_rate']
        self.debug = args['debug']
        self.lambda1 = args['lambda1']
        self.lambda2 = args['lambda2']
        self.lambda3 = args['lambda3']

        self.best_dev_acc = .0
        self.best_test_acc = .0
        self.best_test_rmse = .0

        # initializers for parameters
        self.w_init = xavier()
        self.b_init = tf.initializers.zeros()
        self.e_init = xavier()

        # embeddings in the model
        self.wrd_emb = wrd_emb
        self.usr_emb = var('usr_emb', [self.usr_cnt, self.emb_dim], self.e_init)
        self.prd_emb = var('prd_emb', [self.prd_cnt, self.emb_dim], self.e_init)
        self.embeddings = [self.wrd_emb, self.usr_emb, self.prd_emb]

    def dnsc(self, x, max_sen_len, max_doc_len, sen_len, doc_len, identities):
        x = tf.reshape(x, [-1, max_sen_len, self.emb_dim])
        sen_len = tf.reshape(sen_len, [-1])

        def lstm(inputs, sequence_length, hidden_size, scope):
            cell_fw = tf.nn.rnn_cell.LSTMCell(
                hidden_size // 2, forget_bias=0., initializer=xavier())
            cell_bw = tf.nn.rnn_cell.LSTMCell(
                hidden_size // 2, forget_bias=0., initializer=xavier())
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=inputs,
                sequence_length=sequence_length,
                dtype=tf.float32,
                scope=scope)
            outputs = tf.concat(outputs, axis=2)
            return outputs, state

        with tf.variable_scope('sentence_layer'):
            # lstm_outputs, _state = lstm(x, sen_len, self.hidden_size, 'lstm')
            # lstm_outputs = tf.reshape(lstm_outputs, [-1, max_sen_len, self.hidden_size])
            lstm_bkg, _state = lstm(x, sen_len, self.hidden_size, 'lstm_bkg')
            lstm_bkg = tf.reshape(lstm_bkg,
                                  [-1, max_sen_len, self.hidden_size])
            lstm_outputs = lstm_bkg

            sen_bkg = [
                tf.reshape(
                    tf.tile(bkg[:, None, :], (1, max_doc_len, 1)),
                    (-1, self.hidden_size)) for bkg in identities
            ]
            sen_bkg = hop('attention', lstm_outputs, lstm_bkg, sen_bkg[0],
                          sen_bkg[1:], sen_len, max_sen_len, '')
        outputs = tf.reshape(sen_bkg, [-1, max_doc_len, self.hidden_size])

        with tf.variable_scope('document_layer'):
            # lstm_outputs, _state = lstm(outputs, doc_len, self.hidden_size, 'lstm')
            lstm_bkg, _state = lstm(outputs, doc_len, self.hidden_size,
                                    'lstm_bkg')
            lstm_outputs = lstm_bkg

            doc_bkg = [i for i in identities]
            doc_bkg = hop('attention', lstm_outputs, lstm_bkg, doc_bkg[0],
                          doc_bkg[1:], doc_len, max_doc_len, '')
        outputs = doc_bkg

        return outputs

    def build(self, input_map):
        # get the inputs
        with tf.variable_scope('inputs'):
            usrid, prdid, input_x, sen_len, doc_len = \
                (input_map['usr'], input_map['prd'],
                 input_map['content'],
                 input_map['sen_len'], input_map['doc_len'])

            usr = lookup(self.usr_emb, usrid, name='cur_usr_embedding')
            prd = lookup(self.prd_emb, prdid, name='cur_prd_embedding')
            input_x = lookup(self.wrd_emb, input_x, name='cur_wrd_embedding')

        logits, d_hats = [], []
        for scope, identities in zip(['user_block', 'product_block'],
                                     [[usr], [prd]]):
            with tf.variable_scope(scope):
                logit = self.dnsc(input_x, self.max_sen_len, self.max_doc_len,
                                  sen_len, doc_len, identities)
                logits.append(logit)

                with tf.variable_scope('result'):
                    d_hats.append(
                        tf.layers.dense(
                            logit,
                            self.cls_cnt,
                            kernel_initializer=self.w_init,
                            bias_initializer=self.b_init))

        with tf.variable_scope('result'):
            logits = tf.concat(logits, axis=1, name='huapa_output')
            d_hat = tf.layers.dense(
                logits,
                self.cls_cnt,
                kernel_initializer=self.w_init,
                bias_initializer=self.b_init)

        return d_hat
