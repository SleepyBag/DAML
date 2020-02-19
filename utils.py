import math
from pathlib import Path
from tqdm import tqdm
import data
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import ops

root = Path('.')

def run_set(sess, step_cnt, sealed_metrics, ops=[], global_step=None):
    """
    sealed_metrics, ops are two lists of tuple(variable, type)
    variable is a tensor
    type is one of SUM, ALL, MEAN, LAST, NONE
    """
    sealed_metrics = [(a, 'SUM') if not isinstance(a, tuple) else a
                      for a in sealed_metrics]
    metrics = sealed_metrics + ops
    pgb = tqdm(range(step_cnt), leave=False, dynamic_ncols=True)
    ans = []
    for metric, method in metrics:
        if method == 'ALL':
            ans.append([])
        elif method == 'MEAN' or method == 'SUM':
            ans.append(0)
        elif method == 'LAST' or method == 'NONE':
            ans.append(None)
    methods = [metric[1] for metric in metrics]
    metrics = [metric[0] for metric in metrics]
    for _ in pgb:
        cur_step = sess.run(global_step)
        pgb.set_description(str(cur_step))
        cur_metrics = sess.run(metrics)
        for i, metric, method in zip(range(len(ans)), cur_metrics, methods):
            if method == 'ALL':
                ans[i].append(metric)
            elif method == 'MEAN':
                ans[i] += metric / step_cnt
            elif method == 'SUM':
                ans[i] += metric
            elif method == 'LAST':
                ans[i] = metric
            elif method == 'NONE':
                pass
    return [ans[:len(sealed_metrics)]] + ans[len(sealed_metrics):]


def load_data(dataset, drop, emb_dim, batch_size, max_doc_len,
              max_sen_len, repeat, split_by_period, shuffle_train=True):
    # Load data
    print("Loading data...")
    datasets = [str(root / 'data' / dataset / s)
        for s in ['train.ss', 'dev.ss', 'test.ss']
    ]
    tfrecords = [str(root / 'data' / dataset / 'tfrecords' / s)
        for s in ['train.tfrecord', 'dev.tfrecord', 'test.tfrecord']
    ]
    stats_filename = str(root / 'data' / dataset / 'stats' / ('stats.txt' + str(drop)))
    embedding_filename = 'data/embedding_imdb_yelp13_elc_cd_clt.txt'
    # if dataset in ['yelp13', 'imdb']:
    #     embedding_filename = 'data/embedding_imdb_yelp13.txt'
    # elif dataset in ['cd', 'elc']:
    #     embedding_filename = 'data/embedding_cd_elc.txt'
    print(embedding_filename)
    text_filename = str(root / 'data' / dataset / 'word2vec_train.ss')
    datasets, lengths, embedding, stats, wrd_dict = data.build_dataset(
        datasets, tfrecords, stats_filename, embedding_filename,
        max_doc_len, max_sen_len, split_by_period,
        emb_dim, text_filename, drop)
    trainset, devset, testset = datasets
    trainlen, devlen, testlen = lengths
    #  trainlen *=  1 - flags.drop
    if repeat:
        trainset = trainset.repeat()
        devset = devset.repeat()
        testset = testset.repeat()
    if shuffle_train:
        trainset = trainset.shuffle(30000)
    devset = devset
    testset = testset
    if batch_size != 1:
        trainset = trainset.batch(batch_size)
        devset = devset.batch(batch_size)
        testset = testset.batch(batch_size)
    print("Data loaded.")
    return embedding, trainset, devset, testset, trainlen, devlen, testlen, stats


def get_step_cnt(datalen, batch_size):
    return int(math.ceil(float(datalen) / batch_size))


def get_variables_to_restore(variables, var_keep_dic):
    variables_to_restore = []
    for v in variables:
        # one can do include or exclude operations here.
        if v.name.split(':')[0] in var_keep_dic:
            print("Variables restored: %s" % v.name)
            variables_to_restore.append(v)

    return variables_to_restore


def get_variable_in_checkpoint_file(checkpoint_path):
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    return list(var_to_shape_map.keys())


def get_train_op(optimizer, loss, wrd_emb, global_step, embedding_lr):
    """
    get the training operation of a model with word embedding,
    the grad of word embedding will be scaled according to embedding_lr
    """
    grads_and_vars = optimizer.compute_gradients(loss)
    capped_grads_and_vars = []

    for grad, v in grads_and_vars:
        if v is wrd_emb:
            grad = tf.IndexedSlices(grad.values * embedding_lr, grad.indices,
                                    grad.dense_shape)
        capped_grads_and_vars.append((grad, v))

    train_op = optimizer.apply_gradients(
        capped_grads_and_vars, global_step=global_step)
    return train_op


class FlipGradientBuilder(object):
    '''Gradient Reversal Layer from https://github.com/pumpikano/tf-dann'''

    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]

        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)

        self.num_calls += 1
        return y


flip_gradient = FlipGradientBuilder()
