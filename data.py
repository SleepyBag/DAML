import os
import csv
from functools import partial
import pandas as pd
import tensorflow as tf
import numpy as np

reading_col_name = ['usr', 'prd', 'rating', 'content']
output_col_name = ['usr', 'prd', 'rating', 'content', 'doc_len', 'sen_len']


def build_dataset(filenames, tfrecords_filenames, stats_filename,
                  embedding_filename, max_doc_len,
                  max_sen_len, split_by_period, emb_dim, text_filename, drop):
    datasets = []
    if not os.path.exists(embedding_filename):
        w2v.word2vec(
            text_filename,
            embedding_filename,
            size=emb_dim,
            binary=0,
            verbose=True)
        #  os.system('vim ' + embedding_filename + ' +%s/ $//g +wqall')
    wrd_dict, wrd_index, embedding = load_embedding(embedding_filename,
                                                    emb_dim)

    tfrecords_filenames = [
        i + str(split_by_period) + str(max_doc_len) + str(max_sen_len) +
        str(drop) for i in tfrecords_filenames
    ]
    stats = {}
    if sum([os.path.exists(i) for i in tfrecords_filenames]) < len(tfrecords_filenames) \
            or not os.path.exists(stats_filename):
        for tfrecords_filename in tfrecords_filenames:
            if os.path.exists(tfrecords_filename):
                os.remove(tfrecords_filename)
        if os.path.exists(stats_filename):
            os.remove(stats_filename)
        # read the data and transform them

        data_frames = [
            pd.read_csv(
                filename, sep='\t\t', names=reading_col_name, engine='python')
            for filename in filenames
        ]

        total_data = pd.concat(data_frames)
        usr = total_data.usr.unique().tolist()
        usr.sort()
        usr_index = {name: index for index, name in enumerate(usr)}
        prd = total_data.prd.unique().tolist()
        prd.sort()
        prd_index = {name: index for index, name in enumerate(prd)}

        stats['usr_cnt'] = len(usr)
        stats['prd_cnt'] = len(prd)
        stats['doc_cnt'] = len(data_frames[0]) + 1

        usr_doc = {}
        prd_doc = {}
        for i, item in data_frames[0].iterrows():
            usr = usr_index[item['usr']]
            prd = prd_index[item['prd']]
            if usr not in usr_doc:
                usr_doc[usr] = []
            usr_doc[usr].append(i)
            if prd not in prd_doc:
                prd_doc[prd] = []
            prd_doc[prd].append(i)

        data_frames[0] = data_process(
            data_frames[0], wrd_index, usr_index, prd_index, max_doc_len,
            max_sen_len, split_by_period, drop, 'INC', usr_doc, prd_doc)
        data_frames[1] = data_process(
            data_frames[1], wrd_index, usr_index, prd_index, max_doc_len,
            max_sen_len, split_by_period, .0, 'ZERO', usr_doc, prd_doc)
        data_frames[2] = data_process(
            data_frames[2], wrd_index, usr_index, prd_index, max_doc_len,
            max_sen_len, split_by_period, .0, 'ZERO', usr_doc, prd_doc)

        # build the dataset
        for filename, tfrecords_filename, data_frame in zip(
                filenames, tfrecords_filenames, data_frames):

            def to_string(x):
                return x.tostring()

            data_frame['content'] = data_frame['content'].transform(to_string)
            # data_frame['polarity'] = data_frame['polarity'].transform(
            #     to_string)
            data_frame['sen_len'] = data_frame['sen_len'].transform(to_string)
            # data_frame['co_doc'] = data_frame['co_doc'].transform(to_string)

            writer = tf.python_io.TFRecordWriter(tfrecords_filename)

            def int64list(value):
                return tf.train.Feature(
                    int64_list=tf.train.Int64List(value=value))

            def byteslist(value):
                return tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=value))

            for item in data_frame.iterrows():
                feature = {
                    'usr': int64list([item[1]['usr']]),
                    'prd': int64list([item[1]['prd']]),
                    'rating': int64list([item[1]['rating']]),
                    'content': byteslist([item[1]['content']]),
                    'sen_len': byteslist([item[1]['sen_len']]),
                    'doc_len': int64list([item[1]['doc_len']]),
                    # 'polarity': byteslist([item[1]['polarity']]),
                    # 'co_doc': byteslist([item[1]['co_doc']]),
                    # 'co_doc_cnt': int64list([item[1]['co_doc_cnt']]),
                    'docid': int64list([item[1]['docid']])
                }

                example = tf.train.Example(
                    features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            writer.close()
            stats[filename + 'len'] = len(data_frame)
            # lengths.append(len(data_frame))

        stats_file = csv.writer(open(stats_filename, 'w'))
        # print('usr_cnt: %d, prd_cnt: %d' % (usr_cnt, prd_cnt))
        for key, val in stats.items():
            stats_file.writerow([key, val])

    def transform_example(example):
        def int64_feature():
            return tf.FixedLenFeature(
                shape=(), dtype=tf.int64, default_value=None)

        def string_feature():
            return tf.FixedLenFeature(
                shape=(), dtype=tf.string, default_value=None)

        dics = {
            'usr': int64_feature(),
            'prd': int64_feature(),
            'rating': int64_feature(),
            'content': string_feature(),
            # 'polarity': string_feature(),
            'sen_len': string_feature(),
            'doc_len': int64_feature(),
            # 'co_doc': string_feature(),
            # 'co_doc_cnt': int64_feature(),
            'docid': int64_feature()
        }

        ans = tf.parse_single_example(example, dics)
        ans['content'] = tf.decode_raw(ans['content'], tf.int64)
        # ans['polarity'] = tf.decode_raw(ans['polarity'], tf.int64)
        ans['sen_len'] = tf.decode_raw(ans['sen_len'], tf.int64)
        # ans['co_doc'] = tf.decode_raw(ans['co_doc'], tf.int64)
        return ans

    for key, val in csv.reader(open(stats_filename)):
        stats[key] = int(val)
    for tfrecords_filename in tfrecords_filenames:
        dataset = tf.data.TFRecordDataset(tfrecords_filename)
        dataset = dataset.map(transform_example)
        datasets.append(dataset)

    lengths = [stats[filename + 'len'] for filename in filenames]
    return datasets, lengths, embedding.values, stats, wrd_dict


# load an embedding file
def load_embedding(filename, emb_dim):
    try:
        emb_col_name = ['wrd'] + [i for i in range(emb_dim + 1)]
        data_frame = pd.read_csv(
            filename, sep=' ', header=0, names=emb_col_name)
    except pd.errors.ParserError:
        emb_col_name = ['wrd'] + [i for i in range(emb_dim)]
        data_frame = pd.read_csv(
            filename, sep=' ', header=0, names=emb_col_name)
    data_frame = data_frame.sort_values('wrd')
    embedding = data_frame[emb_col_name[1: emb_dim + 1]]
    wrd_dict = data_frame['wrd'].tolist()
    wrd_index = {s: i for i, s in enumerate(wrd_dict)}
    return wrd_dict, wrd_index, embedding


# transform a sentence into indices
def sentence_transform(document, wrd_index, max_doc_len, max_sen_len,
                       split_by_period):
    if split_by_period:
        sentence_index = np.zeros((max_doc_len, max_sen_len), dtype=np.int)
        for i, sentence in enumerate(document):
            if i >= max_doc_len:
                break
            j = 0
            for wrd in sentence:
                if j >= max_sen_len:
                    break
                if wrd in wrd_index:
                    sentence_index[i][j] = wrd_index[wrd]
                    j += 1
    else:
        sentence_index = np.zeros((max_doc_len * max_sen_len, ), dtype=np.int)
        i = 0
        for wrd in document:
            if i == max_doc_len * max_sen_len:
                break
            if wrd in wrd_index:
                sentence_index[i] = wrd_index[wrd]
                i += 1
        sentence_index = sentence_index.reshape((max_doc_len, max_sen_len))
    return sentence_index


def split_paragraph(paragraph, split_by_period):
    if split_by_period:
        sentences = paragraph.split('<sssss>')
        for i, _ in enumerate(sentences):
            sentences[i] = sentences[i].split()
    else:
        sentences = paragraph.split()
    return sentences


def data_process(df, wrd_index, usr_index, prd_index, max_doc_len, max_sen_len,
                 split_by_period, drop, docid_type, usr_doc, prd_doc):
    """
    docid is fixed to two values: 'INC' and 'ZERO'
    """
    # count contents' length
    df.content = df.content.apply(
        partial(split_paragraph, split_by_period=split_by_period))
    if drop != 0.:
        df.rating = df.rating.apply(lambda x: x * np.random.choice(
            [0, 1], p=[drop, 1 - drop]))
        df = df[df.rating.isin(range(1, 100))]
    df.rating = df.rating - 1
    # df['max_sen_len'] = df['sen_len'].transform(lambda sen_len: max(sen_len))

    # transform users and products to indices
    df.usr = df.usr.map(usr_index)
    df.prd = df.prd.map(prd_index)
    print('Users and products indexed.')

    # transform contents into indices
    # df['polarity'] = df.content.transform(
    #     partial(
    #         sentence_to_polarity,
    #         max_doc_len=max_doc_len,
    #         max_sen_len=max_sen_len,
    #         split_by_period=split_by_period))
    df.content = df.content.transform(
        partial(
            sentence_transform,
            wrd_index=wrd_index,
            max_doc_len=max_doc_len,
            max_sen_len=max_sen_len,
            split_by_period=split_by_period))
    df['sen_len'] = df.content.transform(lambda i: np.count_nonzero(i, axis=1))
    df['doc_len'] = df.sen_len.transform(np.count_nonzero)
    print('Contents indexed.')

    # add docid intot data
    if docid_type == 'INC':
        df['docid'] = df.index + 1
    elif docid_type == 'ZERO':
        df['docid'] = 0
    #  df['co_doc'] = df.

    # df['co_doc'] = df.apply(
    #     lambda row: query_co_doc(row['usr'], row['prd'], row['docid'], usr_doc,
    #                              prd_doc),
    #     axis=1)
    # df['co_doc_cnt'] = df.co_doc.transform(np.count_nonzero)
    # print('co_doc counted')
    return df
