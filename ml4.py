from math import sqrt
import tensorflow as tf
from tensorflow import constant as const
from tensorflow import stop_gradient as stop
from tensorflow.contrib.layers import xavier_initializer as xavier
from layers.nscla import NSCLA
from utils import flip_gradient


class ML4(object):
    def __init__(self, args):
        self.embedding = args['embedding']
        self.wrd_emb = const(self.embedding, name='wrd_emb', dtype=tf.float32)
        self.model_cnt = 4
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

        self.model = []
        for i in range(self.model_cnt):
            with tf.variable_scope(f'model{i}'):
                self.model.append(NSCLA(args, self.wrd_emb))

        # initializers for parameters
        self.w_init = xavier()
        self.b_init = tf.initializers.zeros()
        self.e_init = xavier()

    def build(self, data_iter, global_step):
        """
        build the whole model from a input iterator,
        global step is not used, it's here just to fit the training code
        """
        # define aliases
        ones_like = lambda x: tf.to_float(tf.ones_like(x))
        zeros_like = lambda x: tf.to_float(tf.zeros_like(x))
        ssce = tf.nn.sparse_softmax_cross_entropy_with_logits
        sce = tf.nn.softmax_cross_entropy_with_logits
        dense = lambda feature, cls_cnt: tf.layers.dense(feature, cls_cnt, kernel_initializer=self.w_init, bias_initializer=self.b_init)

        input_map = data_iter.get_next()

        rating = input_map['rating']
        hard_label = tf.one_hot(rating, self.cls_cnt)
        task_label = input_map['task_label']

        features = []
        logits = []
        soft_logits = []
        soft_labels = []
        # build the body of models
        for i in range(self.model_cnt):
            with tf.variable_scope(f'model{i}'):
                features.append(self.model[i].build(input_map))
                logits.append(dense(features[-1], self.cls_cnt))
                soft_labels.append(tf.nn.softmax(stop(logits[-1]) / self.temperature))
        for i in range(self.model_cnt):
            soft_logits.append([])
            for _ in range(self.model_cnt - 1):
                soft_logits[-1].append(dense(features[i], self.cls_cnt))

        # obtain the loss
        def get_loss(logits, soft_logits, labels, soft_labels, task_label):
            loss_truth = sce(logits=logits, labels=labels)
            loss_truth = tf.where(tf.equal(task_label, 0), loss_truth, zeros_like(loss_truth))
            for slo, sla in zip(soft_logits, soft_labels):
                loss_soft = sce(logits=slo, labels=sla)
                loss_soft = tf.where(tf.equal(task_label, 0), zeros_like(loss_soft), loss_soft)
                loss_truth += self.align_rate * loss_soft
            return loss_truth

        def get_loss_adv(feature, task_label):
            feature = flip_gradient(feature, 0.005)
            task_logit = tf.layers.dense(feature, self.task_cnt)
            task_logit_dis = task_logit
            loss_adv = ssce(logits=task_logit, labels=task_label)
            return loss_adv, task_logit_dis

        losses = []
        losses_adv = []
        task_logits_dis = []
        for i in range(self.model_cnt):
            with tf.variable_scope(f"loss{i}"):
                losses.append(get_loss(logits[i], soft_logits[i], hard_label,
                                       [sl for j, sl in enumerate(soft_labels) if i != j],
                                       task_label))
            with tf.variable_scope(f"loss_adv{i}"):
                cur_loss_adv, cur_task_logit_dis = get_loss_adv(features[i], task_label)
                losses_adv.append(cur_loss_adv)
                task_logits_dis.append(cur_task_logit_dis)

        total_loss = sum(losses) + sum(losses_adv)
        total_loss = tf.reduce_sum(total_loss)
        self.loss = total_loss

        # metrics
        preds = []
        correct_nums = []
        correct_preds = []
        mses = []
        for i in range(self.model_cnt):
            preds.append(tf.argmax(logits[i], 1))
            correct_preds.append(tf.equal(preds[i], rating))
            correct_nums.append(tf.reduce_sum(tf.to_int32(correct_preds[i])))
            mses.append(tf.reduce_sum(tf.square(preds[i] - rating)))

        task_pred = []
        task_correct_pred = []
        task_correct_nums = []
        with tf.variable_scope("metrics"):
            for i in range(self.model_cnt):
                task_pred.append(tf.argmax(task_logits_dis[i], 1))
                task_correct_pred.append(tf.equal(task_pred[i], task_label))
                task_correct_nums.append(tf.reduce_sum(tf.to_int32(task_correct_pred[i])))

        return [(total_loss, 'SUM')] + [(mse, 'SUM') for mse in mses] + \
                [(correct_num, 'SUM') for correct_num in correct_nums] + \
                [(task_correct_num, 'SUM') for task_correct_num in task_correct_nums]

    def output_metrics(self, metrics, data_length):
        loss = metrics[0]
        mses = metrics[1: 1 + self.model_cnt]
        correct_nums = metrics[1 + self.model_cnt: 1 + 2 * self.model_cnt]
        task_correct_nums = metrics[1 + 2 * self.model_cnt: 1 + 3 * self.model_cnt]
        info = ''
        for i, (mse, correct_num, task_correct_num) in enumerate(zip(mses, correct_nums, task_correct_nums)):
            info += f'M{i}: Loss= {loss / data_length:.3f}, '
            info += f'RMSE= {sqrt(float(mse) / data_length):.3f}, '
            info += f'Acc= {float(correct_num) / data_length:.3f} '
            info += f'DiscriAcc= {float(task_correct_num) / data_length:.3f}\n'
        return info

    def record_metrics(self, dev_metrics, test_metrics, devlen, testlen):
        dev_correct_nums = dev_metrics[1 + self.model_cnt: 1 + 2 * self.model_cnt]
        test_mses = test_metrics[1: 1 + self.model_cnt]
        test_correct_nums = test_metrics[1 + self.model_cnt: 1 + 2 * self.model_cnt]
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
