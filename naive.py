from math import sqrt
import tensorflow as tf
from tensorflow import constant as const
from tensorflow import stop_gradient as stop
from tensorflow.contrib.layers import xavier_initializer as xavier
from layers.nscla import NSCLA
from utils import flip_gradient


class NAIVE(object):
    def __init__(self, args):
        self.embedding = args['embedding']
        self.wrd_emb = const(self.embedding, name='wrd_emb', dtype=tf.float32)
        self.model = NSCLA(args, self.wrd_emb)
        self.l2_rate = args['l2_rate']
        self.cls_cnt = args['cls_cnt']
        self.embedding_lr = args['embedding_lr']
        self.temperature = args['temperature']
        self.align_rate = args['align_rate']
        self.task_cnt = args['task_cnt']
        self.best_test_acc = 0.
        self.best_dev_acc = 0.
        self.best_test_rmse = 0.
        self.hidden_size = args['emb_dim']

        # initializers for parameters
        self.w_init = xavier()
        self.b_init = tf.initializers.zeros()
        self.e_init = xavier()

    def build(self, data_iter):
        input_map = data_iter.get_next()
        self.input_map = input_map
        input_y = input_map['rating']
        self.input_y = input_y
        task_label = input_map['task_label']

        ssce = tf.nn.sparse_softmax_cross_entropy_with_logits
        feature = self.model.build(input_map)
        self.feature = feature
        # this part is trained for the unlabeled dataset which could only leverage the shared part of the model
        with tf.variable_scope("loss_share"):
            #  feature = stop(feature_share)
            logit = tf.layers.dense(feature, self.cls_cnt, kernel_initializer=self.w_init, bias_initializer=self.b_init)
            loss = ssce(logits=logit, labels=input_y)

        total_loss = tf.reduce_sum(loss)
        self.loss = total_loss

        with tf.variable_scope("metrics"):
            pred = tf.argmax(logit, 1)
            correct_pred = tf.equal(pred, input_y)
            mse = tf.reduce_sum(tf.square(pred - input_y))
            correct_num = tf.reduce_sum(tf.to_int32(correct_pred))

        return [(total_loss, 'SUM'), (mse, 'SUM'), (correct_num, 'SUM')]

    def output_metrics(self, metrics, data_length):
        loss, mse, correct_num = metrics
        info = 'Mtotal: Loss = %.3f, RMSE = %.3f, Acc = %.3f\n' % \
            (loss / data_length, sqrt(float(mse) / data_length), float(correct_num) / data_length)
        return info

    def record_metrics(self, dev_metrics, test_metrics, devlen, testlen):
        dev_loss, dev_mse, dev_correct_num = dev_metrics
        test_loss, test_mse, test_correct_num = test_metrics
        dev_accuracy = float(dev_correct_num) / devlen
        test_accuracy = float(test_correct_num) / testlen
        test_rmse = sqrt(float(test_mse) / testlen)
        if dev_accuracy > self.best_dev_acc:
            self.best_dev_acc = dev_accuracy
            self.best_test_acc = test_accuracy
            self.best_test_rmse = test_rmse
            info = 'NEW best dev acc: %.3f, NEW best test acc: %.3f, NEW best test RMSE: %.3f' % \
                (self.best_dev_acc, self.best_test_acc, self.best_test_rmse)
        else:
            info = 'best dev acc: %.3f, best test acc: %.3f, best test RMSE: %.3f' % \
                (self.best_dev_acc, self.best_test_acc, self.best_test_rmse)
        return info

    def train(self, optimizer, global_step):
        grads_and_vars = optimizer.compute_gradients(self.loss)
        capped_grads_and_vars = []

        for grad, v in grads_and_vars:
            if v is self.model.wrd_emb:
                grad = tf.IndexedSlices(grad.values * self.embedding_lr,
                                        grad.indices, grad.dense_shape)
            capped_grads_and_vars.append((grad, v))

        train_op = optimizer.apply_gradients(
            capped_grads_and_vars, global_step=global_step)
        return train_op
