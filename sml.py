from math import sqrt
import tensorflow as tf
from tensorflow import constant as const
from tensorflow import stop_gradient as stop
from tensorflow.contrib.layers import xavier_initializer as xavier
from layers.nscla import NSCLA
from utils import flip_gradient


class SML(object):
    def __init__(self, args):
        self.embedding = args['embedding']
        self.wrd_emb = const(self.embedding, name='wrd_emb', dtype=tf.float32)
        with tf.variable_scope('model1'):
            self.model1 = NSCLA(args, self.wrd_emb)
        with tf.variable_scope('model2'):
            self.model2 = NSCLA(args, self.wrd_emb)
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

    def build(self, data_iter1, data_iter2, global_step):
        """
        build the whole model from a input iterator,
        global step is not used, it's here just to align with the training code
        """
        # define aliases
        ones_like = lambda x: tf.to_float(tf.ones_like(x))
        zeros_like = lambda x: tf.to_float(tf.zeros_like(x))
        ssce = tf.nn.sparse_softmax_cross_entropy_with_logits
        sce = tf.nn.softmax_cross_entropy_with_logits

        # parse the input
        if data_iter1 is data_iter2:
            input_map1 = data_iter1.get_next()
            input_map2 = input_map1
        else:
            input_map1 = data_iter1.get_next()
            input_map2 = data_iter2.get_next()

        rating1 = input_map1['rating']
        hard_label1 = tf.one_hot(rating1, self.cls_cnt)
        task_label1 = input_map1['task_label']

        rating2 = input_map2['rating']
        hard_label2 = tf.one_hot(rating2, self.cls_cnt)
        task_label2 = input_map2['task_label']

        self.input_y = rating1

        # build the body of models
        with tf.variable_scope('model1'):
            feature1 = self.model1.build(input_map1)
        with tf.variable_scope('model2'):
            feature2 = self.model2.build(input_map2)
        self.feature = feature1
        logit1 = tf.layers.dense(feature1, self.cls_cnt, kernel_initializer=self.w_init, bias_initializer=self.b_init)
        soft_label1 = tf.nn.softmax(tf.stop_gradient(logit1) / self.temperature)
        logit2 = tf.layers.dense(feature2, self.cls_cnt, kernel_initializer=self.w_init, bias_initializer=self.b_init)
        soft_label2 = tf.nn.softmax(tf.stop_gradient(logit2) / self.temperature)

        # obtain the loss
        def get_loss(logits, labels, soft_labels, task_label):
            soft_labels = stop(soft_labels)
            loss_truth = sce(logits=logits, labels=labels)
            loss_truth = tf.where(tf.equal(task_label, 0), loss_truth, zeros_like(loss_truth))
            loss_soft = sce(logits=logits, labels=soft_labels)
            loss_soft = tf.where(tf.equal(task_label, 0), zeros_like(loss_soft), loss_soft)
            loss_truth = loss_truth + self.align_rate * loss_soft
            return loss_truth

        def get_loss_adv(feature, task_label):
            feature = flip_gradient(feature, 0.005)
            task_logit = tf.layers.dense(feature, self.task_cnt, kernel_initializer=self.w_init, bias_initializer=self.b_init)
            task_logit_dis = task_logit
            loss_adv = ssce(logits=task_logit, labels=task_label)
            return loss_adv, task_logit_dis

        with tf.variable_scope("loss1"):
            loss1 = get_loss(logit1, hard_label1, soft_label2, task_label1)
        with tf.variable_scope("loss2"):
            loss2 = get_loss(logit2, hard_label2, soft_label1, task_label2)
        with tf.variable_scope('loss_adv1'):
            loss_adv1, task_logit_dis1 = get_loss_adv(feature1, task_label1)
        with tf.variable_scope('loss_adv2'):
            loss_adv2, task_logit_dis2 = get_loss_adv(feature2, task_label2)

        self.total_loss1 = tf.reduce_sum(loss1 + loss_adv1)
        self.total_loss2 = tf.reduce_sum(loss2 + loss_adv2)
        total_loss = loss1 + loss2 + loss_adv1 + loss_adv2
        total_loss = tf.reduce_sum(total_loss)
        self.loss = total_loss

        # metrics
        pred1 = tf.argmax(logit1, 1)
        pred2 = tf.argmax(logit2, 1)
        same_pred = tf.equal(pred1, pred2)
        same_num = tf.reduce_sum(tf.to_int32(same_pred))
        correct_pred1 = tf.equal(pred1, rating1)
        correct_pred2 = tf.equal(pred2, rating2)
        correct_num1 = tf.reduce_sum(tf.to_int32(correct_pred1))
        correct_num2 = tf.reduce_sum(tf.to_int32(correct_pred2))
        mse1 = tf.reduce_sum(tf.square(pred1 - rating1))
        mse2 = tf.reduce_sum(tf.square(pred2 - rating2))

        with tf.variable_scope("metrics"):
            task_pred1 = tf.argmax(task_logit_dis1, 1)
            task_pred2 = tf.argmax(task_logit_dis2, 1)
            task_correct_pred1 = tf.equal(task_pred1, task_label1)
            task_correct_pred2 = tf.equal(task_pred2, task_label2)
            task_correct_num1 = tf.reduce_sum(tf.to_int32(task_correct_pred1))
            task_correct_num2 = tf.reduce_sum(tf.to_int32(task_correct_pred2))

        return [(total_loss, 'SUM'), (mse1, 'SUM'), (mse2, 'SUM'),
                (correct_num1, 'SUM'), (correct_num2, 'SUM'),
                (task_correct_num1, 'SUM'), (task_correct_num2, 'SUM'),
                (same_num, 'SUM')]

    def output_metrics(self, metrics, data_length):
        loss, mse1, mse2, correct_num1, correct_num2, task_correct_num1, task_correct_num2, same_num = metrics
        info = 'M1: Loss= %.3f, RMSE= %.3f, Acc= %.3f ' % \
            (loss / data_length, sqrt(float(mse1) / data_length), float(correct_num1) / data_length)
        info += 'DiscriAcc= %.3f\n' % \
            (float(task_correct_num1) / data_length)
        info += 'M2: Loss= %.3f, RMSE= %.3f, Acc= %.3f ' % \
            (loss / data_length, sqrt(float(mse2) / data_length), float(correct_num2) / data_length)
        info += 'DiscriAcc= %.3f\n' % \
            (float(task_correct_num2) / data_length)
        info += 'Align rate= %.3f\n' % (same_num / data_length)
        return info

    def record_metrics(self, dev_metrics, test_metrics, devlen, testlen):
        dev_correct_nums = [None] * 2
        test_mses = [None] * 2
        test_correct_nums = [None] * 2
        _, _, _, dev_correct_nums[0], dev_correct_nums[1], _, _, _ = dev_metrics
        _, test_mses[0], test_mses[1], test_correct_nums[0], test_correct_nums[1], _, _, _ = test_metrics
        info = ''
        for dev_correct_num, test_mse, test_correct_num in zip(dev_correct_nums, test_mses, test_correct_nums):
            dev_accuracy = float(dev_correct_num) / devlen
            test_accuracy = float(test_correct_num) / testlen
            test_rmse = sqrt(float(test_mse) / testlen)
            if dev_accuracy > self.best_dev_acc:
                self.best_dev_acc = dev_accuracy
                self.best_test_acc = test_accuracy
                self.best_test_rmse = test_rmse
                curinfo = 'NEW best dev acc: %.3f, NEW best test acc: %.3f, NEW best test RMSE: %.3f' % \
                    (self.best_dev_acc, self.best_test_acc, self.best_test_rmse)
            else:
                curinfo = 'best dev acc: %.3f, best test acc: %.3f, best test RMSE: %.3f' % \
                    (self.best_dev_acc, self.best_test_acc, self.best_test_rmse)
            if not info or 'NEW' in curinfo:
                info = curinfo
        return info
