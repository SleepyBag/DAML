# -*- coding: utf-8 -*-
# author: Xue Qianming
import os
import time
import tensorflow as tf
import numpy as np
from colored import fg, stylize
from utils import run_set, load_data, get_step_cnt, get_variable_in_checkpoint_file, get_variables_to_restore, get_train_op

tf.logging.set_verbosity(tf.logging.ERROR)

# def logger
def log(s):
    print(s)

#                                        _                                   _
#  _ __   __ _ _ __ __ _ _ __ ___   __ _| |_ ___ _ __ ___   _ __   __ _ _ __| |_
# | '_ \ / _` | '__/ _` | '_ ` _ \ / _` | __/ _ \ '__/ __| | '_ \ / _` | '__| __|
# | |_) | (_| | | | (_| | | | | | | (_| | ||  __/ |  \__ \ | |_) | (_| | |  | |_
# | .__/ \__,_|_|  \__,_|_| |_| |_|\__,_|\__\___|_|  |___/ | .__/ \__,_|_|   \__|
# |_|                                                      |_|

# fixed hyperparams (they doesn't worth tuning)
DROP_RATE = 0.
SPLIT_BY_PERIOD = True
TASK_CNT = 2

params = {
    'debug_params': [('debug', False, 'Whether to debug or not'),
                     ('check', False, 'Whether to make a checkpoint')],
    'data_params': [('cls_cnt', 5, "Numbers of class"),
                    ('dataset', 'yelp13', "The dataset"),
                    ('unlabeled_dataset', 'imdb', "The unlabeled dataset")],
    'model_chooing': [('model', 'dhuapa', 'Model to train')],
    'model_hyperparam':
    [("emb_dim", 200, "size of embedding"), ("hidden_size", 100,
                                             "hidden_size"),
     ('max_sen_len', 50, 'max number of tokens per sentence'),
     ('max_doc_len', 40, 'max number of sentences per document'),
     ('max_co_doc_cnt', 1000, 'max number of the sum of U(d) and P(d)'),
     ('sen_aspect_cnt', 1, 'max number of tokens per sentence'),
     ('doc_aspect_cnt', 1, 'max number of tokens per document'),
     ('sen_hop_cnt', 1, 'layers of memnet in sentence layer'),
     ('doc_hop_cnt', 1, 'layers of memnet in document layer'),
     ('hop_cnt', 1, 'layers of memnet in the end'),
     ('temperature', 1., 'the temperature of soft labels'),
     ("lr", .001, "Learning rate"), ("l2_rate", 0.,
                                     "rate of l2 regularization"),
     ("embedding_lr", 1e-5, "embedding learning rate"),
     ("emb_threshold", 1., "embedding learning rate"),
     ("align_rate", 1., "the rate of mutual aligning loss"),
     ("lambda1", 1., "proportion of the total loss"),
     ("lambda2", .0, "proportion of the loss of user block"),
     ("lambda3", .0, "proportion of the loss of product block"),
     ("adv_type", 'gaussian', "type of adversarial noise"),
     ("adv_scale", 0.1, "scale of adversarial noise"),
     ("adv_rate", 0.1, "rate of adversarial loss"),
     ("bilstm", True, "use biLSTM or LSTM"),
     ("split_by_period", True,
      "whether to split the document by sentences or fixed length")],
    'training_params': [("batch_size", 100, "Batch Size"),
                        ("epoch_cnt", 10, "Number of training epochs"),
                        ("checkpoint", '', "checkpoint to restore params"),
                        ("training_method", 'adam',
                         'Method chose to tune the weights')],
    'misc_params':
    [("allow_soft_placement", True, "Allow device soft device placement"),
     ("log_device_placement", False, "Log placement of ops on devices")]
}

for param_collection in list(params.values()):
    for param_name, default, description in param_collection:
        param_type = type(default)
        if param_type is int:
            tf.flags.DEFINE_integer(param_name, default, description)
        elif param_type is float:
            tf.flags.DEFINE_float(param_name, default, description)
        elif param_type is str:
            tf.flags.DEFINE_string(param_name, default, description)
        elif param_type is bool:
            tf.flags.DEFINE_boolean(param_name, default, description)

flags = tf.flags.FLAGS

# force to parse flags
_ = flags.batch_size
# print params
log("\nParameters:")
for attr, value in sorted(flags.__flags.items()):
    log(("{}={}".format(attr.upper(), value.value)))
log("")

#      _       _                 _                      _
#   __| | __ _| |_ __ _ ___  ___| |_   _ __   __ _ _ __| |_
#  / _` |/ _` | __/ _` / __|/ _ \ __| | '_ \ / _` | '__| __|
# | (_| | (_| | || (_| \__ \  __/ |_  | |_) | (_| | |  | |_
#  \__,_|\__,_|\__\__,_|___/\___|\__| | .__/ \__,_|_|   \__|
#                                     |_|

def temp_load_data(dataset, task_label):
    embedding, trainset, devset, testset, trainlen, devlen, testlen, stats = load_data(
        dataset,
        DROP_RATE,
        flags.emb_dim,
        1,
        flags.max_doc_len,
        flags.max_sen_len,
        repeat=False,
        split_by_period=True)

    def transform(x):
        x['task_label'] = tf.zeros_like(x['usr']) + task_label
        if dataset == 'imdb':
            x['rating'] //= 2
        return x

    trainset = trainset.map(transform)
    devset = devset.map(transform)
    testset = testset.map(transform)

    return embedding, trainset, devset, testset, trainlen, devlen, testlen, stats


embedding, trainset, devset, testset, trainlen, devlen, testlen, stats = \
    temp_load_data(flags.dataset, 0)
unlabeled_embedding, unlabeled_trainset, unlabeled_devset, unlabeled_testset, unlabeled_trainlen, unlabeled_devlen, unlabeled_testlen, unlabeled_stats = \
    temp_load_data(flags.unlabeled_dataset, 1)
devset = devset.batch(flags.batch_size)
testset = testset.batch(flags.batch_size)
unlabeled_devset = unlabeled_devset.batch(flags.batch_size)
unlabeled_testset = unlabeled_testset.batch(flags.batch_size)
#  trainset = trainset.concatenate(unlabeled_trainset).batch(flags.batch_size).shuffle(30000)
choice_dataset = tf.data.Dataset.range(2).repeat()
labeled_trainset = trainset
trainset = tf.contrib.data.choose_from_datasets([trainset.repeat(), unlabeled_trainset.repeat()],
                                                choice_dataset)
trainset = trainset.batch(flags.batch_size)
labeled_trainlen = trainlen
# trainlen += unlabeled_trainlen
trainlen = min(trainlen, unlabeled_trainlen) * 2
labeled_trainset = labeled_trainset.batch(flags.batch_size)
unlabeled_trainset = unlabeled_trainset.batch(flags.batch_size)

# create data iterators
data_iter = tf.data.Iterator.from_structure(
    trainset.output_types, output_shapes=trainset.output_shapes)
traininit = data_iter.make_initializer(trainset)
labeled_traininit = data_iter.make_initializer(labeled_trainset)
unlabeled_traininit = data_iter.make_initializer(unlabeled_trainset)
devinit = data_iter.make_initializer(devset)
testinit = data_iter.make_initializer(testset)
unlabeled_devinit = data_iter.make_initializer(unlabeled_devset)
unlabeled_testinit = data_iter.make_initializer(unlabeled_testset)

#  ___  ___  ___ ___(_) ___  _ __
# / __|/ _ \/ __/ __| |/ _ \| '_ \
# \__ \  __/\__ \__ \ | (_) | | | |
# |___/\___||___/___/_|\___/|_| |_|

# create the session
session_config = tf.ConfigProto(
    allow_soft_placement=flags.allow_soft_placement,
    log_device_placement=flags.log_device_placement)
session_config.gpu_options.allow_growth = True
sess = tf.Session(config=session_config)

#                      _      _                    _
#  _ __ ___   ___   __| | ___| |  _ __   __ _ _ __| |_
# | '_ ` _ \ / _ \ / _` |/ _ \ | | '_ \ / _` | '__| __|
# | | | | | | (_) | (_| |  __/ | | |_) | (_| | |  | |_
# |_| |_| |_|\___/ \__,_|\___|_| | .__/ \__,_|_|   \__|
#                                |_|

# # build the model params
# build the model params
model_params = flags.__dict__['__wrapped'].__dict__['__flags']
model_params = {i: model_params[i].value for i in model_params}
model_params['embedding'] = embedding
model_params['task_cnt'] = TASK_CNT
model_params.update(stats)
exec('from ' + flags.model + ' import ' + flags.model.upper() + ' as model')
model = model(model_params)
os.system(f"figlet -w {os.get_terminal_size().columns} -f slant {flags.model}")

global_step = tf.Variable(0, name="global_step", trainable=False)
# build the model graph
try:
    metrics = model.build(data_iter, global_step)
except TypeError:
    metrics = model.build(data_iter, data_iter, global_step)

# Define Training procedure
if flags.training_method == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(flags.lr)
elif flags.training_method == 'adam':
    optimizer = tf.train.AdamOptimizer(flags.lr)
elif flags.training_method == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(flags.lr, epsilon=1e-6)
train_op_without_wrd_emb = get_train_op(optimizer, model.loss, model.wrd_emb,
                                        global_step, .0)
train_op_with_wrd_emb = get_train_op(optimizer, model.loss, model.wrd_emb,
                                     global_step, flags.embedding_lr)

#                      _      _                              _
#  _ __ ___   ___   __| | ___| |  _ __ ___  ___ ___  _ __ __| | ___ _ __
# | '_ ` _ \ / _ \ / _` |/ _ \ | | '__/ _ \/ __/ _ \| '__/ _` |/ _ \ '__|
# | | | | | | (_) | (_| |  __/ | | | |  __/ (_| (_) | | | (_| |  __/ |
# |_| |_| |_|\___/ \__,_|\___|_| |_|  \___|\___\___/|_|  \__,_|\___|_|

# merge tensorboard summary
summary = None
if flags.debug:
    summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('summary/train', sess.graph)
    # dev_writer = tf.summary.FileWriter('summary/dev', sess.graph)
    # test_writer = tf.summary.FileWriter('summary/test', sess.graph)

if flags.check:
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

#  _       _ _   _       _ _          _   _
# (_)_ __ (_) |_(_) __ _| (_)______ _| |_(_) ___  _ __
# | | '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \
# | | | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
# |_|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|

if flags.checkpoint == '':
    sess.run(tf.global_variables_initializer())
else:
    # restore the params
    checkpoint_path = os.path.join(
        'ckpts', flags.model,
        flags.checkpoint) if '/' not in flags.checkpoint else os.path.join(
            'ckpts', flags.checkpoint)
    global_variables = tf.global_variables()
    var_keep_dic = get_variable_in_checkpoint_file(checkpoint_path)
    variable_to_restore = get_variables_to_restore(global_variables,
                                                   var_keep_dic)
    saver = tf.train.Saver(variable_to_restore)
    saver.restore(sess, 'ckpts/' + flags.checkpoint)

    # initialize other params
    uninitialized_vars = sess.run(tf.report_uninitialized_variables())
    uninitialized_vars = [s.decode() + ':0' for s in uninitialized_vars]
    uninitialized_vars = [
        var for var in tf.global_variables() if var.name in uninitialized_vars
    ]
    init_uninitialized_op = tf.initialize_variables(uninitialized_vars)
    sess.run(init_uninitialized_op)

#                         _
#  _ __ _   _ _ __  _ __ (_)_ __   __ _
# | '__| | | | '_ \| '_ \| | '_ \ / _` |
# | |  | |_| | | | | | | | | | | | (_| |
# |_|   \__,_|_| |_|_| |_|_|_| |_|\__, |
#                                 |___/

train_op = train_op_without_wrd_emb
for epoch in range(flags.epoch_cnt):
    sess.run(traininit)
    # train on trainset
    # trainlen = flags.batch_size * flags.evaluate_every
    # when debugging, summary info is needed for tensorboard
    # cur_trainlen = trainlen if model.best_test_acc < 0.530 \
    #     else flags.evaluate_every * flags.batch_size
    best_test_acc = model.best_test_acc[0] if isinstance(model.best_test_acc, list) \
        else model.best_test_acc
    if train_op is not train_op_with_wrd_emb and best_test_acc > flags.emb_threshold:
        print('Word embedding is going to be trained')
        train_op = train_op_with_wrd_emb

    if summary is not None:
        train_metrics, step, train_summary, _ = run_set(
            sess, get_step_cnt(trainlen, flags.batch_size), metrics,
            [(global_step, 'ALL'), (summary, 'ALL'), (train_op, 'NONE')], global_step)
    else:
        train_metrics, step, _ = run_set(
            sess, get_step_cnt(trainlen, flags.batch_size), metrics,
            [(global_step, 'ALL'), (train_op, 'NONE')], global_step)

     # train_metrics, step, _ = \
     #     run_set(sess, trainlen, metrics, (global_step, train_op, ))

    def print_info(cur_info, set_type, color):
        cur_info = cur_info.split('\n')
        for i, cinfo in enumerate(cur_info):
            if i == 0:
                cinfo = set_type + ' ' + cinfo + ':'
            else:
                cinfo = ' ' * (len(set_type) + 1) + cinfo
            log((stylize(cinfo, fg(color))))

    info = model.output_metrics(train_metrics, trainlen)
    print_info(info, 'Train', 'yellow')

    if summary is not None:
        for i, s in zip(step, train_summary):
            train_writer.add_summary(s, i)
        train_writer.flush()

    def test(metrics, init_op, set_len, set_name, color):
        sess.run(init_op)
        cur_metrics, = run_set(sess, get_step_cnt(
            set_len, flags.batch_size), metrics, global_step=global_step)
        info = model.output_metrics(cur_metrics, set_len)
        print_info(info, set_name, color)
        return cur_metrics

    dev_metrics = test(metrics, devinit, devlen, 'Dev', 'green')
    # test_metrics = test(metrics, testinit, testlen, 'Test', 'red')
    # unlabeled_dev_metrics = test(metrics, unlabeled_devinit,
    #                              unlabeled_devlen, 'UDev', 'green')
    unlabeled_test_metrics = test(metrics, unlabeled_testinit,
                                  unlabeled_testlen, 'UTest', 'red')

    info = model.record_metrics(dev_metrics, unlabeled_test_metrics, devlen, unlabeled_testlen)
    info = 'Epoch %d finished, ' % epoch + info
    log((stylize(info, fg('white'))))

    # if 'NEW' in info:
    #     sess.run(unlabeled_testinit)
    #     feature, label = run_set(sess, get_step_cnt(unlabeled_testlen, flags.batch_size),
    #                              [(model.feature, 'ALL'), (model.input_y, 'ALL')], global_step=global_step)[0]
    #     feature = np.concatenate(feature, axis=0)
    #     label = np.concatenate(label, axis=0)
    #     np.save('feature/' + flags.model + '_x', feature)
    #     np.save('feature/' + flags.model + '_y', label)

    if not 'NEW' in info and epoch >= 10:
        break

    # write a checkpoint
    if flags.check and 'NEW' in info:
        try:
            os.mkdir('ckpts/' + flags.model)
        except:
            pass
        save_path = saver.save(
            sess, 'ckpts/' + flags.model + '/', global_step=step[-1])
        print(('Checkpoint saved to ' + save_path))
