import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xavier
from tensorflow.nn import embedding_lookup as lookup
from layers.attention import attention


def var(name, shape, initializer):
    return tf.get_variable(name, shape=shape, initializer=initializer)


class NSCLA(object):
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
        self.task_cnt = args['task_cnt']

        self.best_dev_acc = .0
        self.best_test_acc = .0
        self.best_test_rmse = .0

        # initializers for parameters
        self.w_init = xavier()
        self.b_init = tf.initializers.zeros()
        self.e_init = xavier()

        # embeddings in the model
        self.wrd_emb = wrd_emb
        self.embeddings = [self.wrd_emb]

    def dnsc(self, x, max_sen_len, max_doc_len, sen_len, doc_len, task_label):
        x = tf.reshape(x, [-1, max_sen_len, self.emb_dim])
        sen_len = tf.reshape(sen_len, [-1])

        def lstm(inputs, sequence_length, hidden_size, scope, init_state):
            init_state_fw, init_state_bw = init_state
            cell_fw = tf.nn.rnn_cell.LSTMCell(
                hidden_size // 2, forget_bias=0., initializer=xavier())
            cell_bw = tf.nn.rnn_cell.LSTMCell(
                hidden_size // 2, forget_bias=0., initializer=xavier())
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                initial_state_fw=init_state_fw,
                initial_state_bw=init_state_bw,
                inputs=inputs,
                sequence_length=sequence_length,
                dtype=tf.float32,
                scope=scope)
            outputs = tf.concat(outputs, axis=2)
            return outputs, state

        with tf.variable_scope('sentence_layer'):
            # lstm_outputs, _state = lstm(x, sen_len, self.hidden_size, 'lstm')
            # lstm_outputs = tf.reshape(lstm_outputs, [-1, max_sen_len, self.hidden_size])
            sen_task_label = tf.reshape(tf.tile(task_label[:, None], [1, max_doc_len]), [-1])
            sen_init_state = tf.get_variable('sen_init_state', [self.task_cnt, 2 * self.hidden_size])
            sen_init_state = tf.nn.embedding_lookup(sen_init_state, sen_task_label)
            sen_init_state_fw = tf.nn.rnn_cell.LSTMStateTuple(sen_init_state[:, :self.hidden_size // 2], sen_init_state[:, self.hidden_size // 2: self.hidden_size])
            sen_init_state_bw = tf.nn.rnn_cell.LSTMStateTuple(sen_init_state[:, self.hidden_size: self.hidden_size * 3 // 2], sen_init_state[:, self.hidden_size * 3 // 2:])
            sen_init_state = (sen_init_state_fw, sen_init_state_bw)
            lstm_bkg, _state = lstm(x, sen_len, self.hidden_size, 'lstm_bkg', sen_init_state)
            lstm_bkg = tf.reshape(lstm_bkg, [-1, max_sen_len, self.hidden_size])
            lstm_outputs = lstm_bkg

            alphas = attention(
                lstm_bkg, [],
                sen_len,
                max_sen_len,
                biases_initializer=self.b_init,
                weights_initializer=self.w_init)
            sen_bkg = alphas @ lstm_outputs
            sen_bkg = tf.reshape(
                sen_bkg, [-1, self.hidden_size], name='new_bkg')
        outputs = tf.reshape(sen_bkg, [-1, max_doc_len, self.hidden_size])
        self.alphas = alphas

        with tf.variable_scope('document_layer'):
            # lstm_outputs, _state = lstm(outputs, doc_len, self.hidden_size, 'lstm')
            doc_task_label = task_label
            doc_init_state = tf.get_variable('doc_init_state', [self.task_cnt, 2 * self.hidden_size])
            doc_init_state = tf.nn.embedding_lookup(doc_init_state, doc_task_label)
            doc_init_state_fw = tf.nn.rnn_cell.LSTMStateTuple(doc_init_state[:, :self.hidden_size // 2], doc_init_state[:, self.hidden_size // 2: self.hidden_size])
            doc_init_state_bw = tf.nn.rnn_cell.LSTMStateTuple(doc_init_state[:, self.hidden_size: self.hidden_size * 3 // 2], doc_init_state[:, self.hidden_size * 3 // 2:])
            doc_init_state = (doc_init_state_fw, doc_init_state_bw)
            lstm_bkg, _state = lstm(outputs, doc_len, self.hidden_size, 'lstm_bkg', doc_init_state)
            lstm_outputs = lstm_bkg

            alphas = attention(
                lstm_bkg, [],
                doc_len,
                max_doc_len,
                biases_initializer=self.b_init,
                weights_initializer=self.w_init)
            doc_bkg = alphas @ lstm_outputs
            doc_bkg = tf.reshape(
                doc_bkg, [-1, self.hidden_size], name='new_bkg')
        outputs = doc_bkg

        return outputs

    def build(self, input_map, seperate_encoder=False):
        # get the inputs
        with tf.variable_scope('inputs'):
            input_x, sen_len, doc_len, task_label = \
                (input_map['content'], input_map['sen_len'],
                 input_map['doc_len'], input_map['task_label'])

            if not seperate_encoder:
                task_label = tf.zeros_like(task_label)
            input_x = lookup(self.wrd_emb, input_x)

        feature = self.dnsc(input_x, self.max_sen_len, self.max_doc_len, sen_len,
                            doc_len, task_label)
        return feature
