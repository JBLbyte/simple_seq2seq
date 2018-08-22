#!/usr/bin/env python
# coding:utf-8


import numpy as np
import time
import datetime
import os
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from utils import Utils
from seq2seq_model import Seq2SeqModel


### hyper parameters
#######################################
lr = 1e-3
epochs = 1000
batch_size = 32
rnn_size = 50
num_layers = 2
encoding_embedding_size = 15
decoding_embedding_size = 15
dropout_keep_prob = 0.7
attention_type='Bahdanau', # or 'Luong'
cell_type = 'lstm'
beam_width = 10

num_checkpoints = 10
evaluate_every = 100
checkpoint_every = 100

use_pre_trained_model = False
model_dir = './model'
checkpoint_file = 'model-5400'
#######################################


### data preprocessing
utils = Utils()
x, y = utils.generate_char_x_y(data_size=10000, seed=314)
data = x + y
id2word_x, word2id_x = utils.extract_character_vocab(x)
id2word_y, word2id_y = utils.extract_character_vocab(y)
x_vocab_size = len(id2word_x)
y_vocab_size = len(id2word_y)

# 将每一行转换成字符id的list
x_ids = [[word2id_x.get(word, word2id_x['<UNK>'])
    for word in sentence] for sentence in x]
y_ids = [[word2id_y.get(word, word2id_y['<UNK>'])
    for word in sentence] + [word2id_y['<EOS>']] for sentence in y]
print(len(x_ids))
print(len(y_ids))
x_train = x_ids[batch_size:]
y_train = y_ids[batch_size:]
data_size_train = len(x_train)
print(x_train[:10])
print(y_train[:10])

x_dev = x_ids[:batch_size]
y_dev = y_ids[:batch_size]
data_size_dev = len(x_dev)
x_dev, y_dev, x_dev_lengths, y_dev_lengths = utils.get_feed_in_data(x_dev, y_dev, word2id_x['<PAD>'], word2id_y['<PAD>'])
print(utils.get_sentence_from_ids(x_dev[0], id2word_x))
print(utils.get_sentence_from_ids(y_dev[0], id2word_y))
input('\nPress enter to start training...')


### load pre-trained model to predict
checkpoint_dir = './log/1534925453/checkpoints'
checkpoint_file = 'model-1000'
checkpoint_file_path = '{}/{}.meta'.format(checkpoint_dir, checkpoint_file)
print(checkpoint_file_path)

saver = tf.train.import_meta_graph('./model/my-model.meta')

# with tf.Session() as sess:

'''
saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

# Access and create placeholders variables and create feed-dict to feed new data
graph = tf.get_default_graph()
input_x_tensor = graph.get_tensor_by_name('input_x_tensor:0')
input_y_tensor = graph.get_tensor_by_name('input_y_tensor:0')
x_sequence_length = graph.get_tensor_by_name('x_sequence_length:0')
y_sequence_length = graph.get_tensor_by_name('y_sequence_length:0')
dropout_keep_prob = graph.get_tensor_by_name('dropout_keep_prob:0')
feed_dict = {
    input_x_tensor: x_dev,
    input_y_tensor: y_dev,
    x_sequence_length: x_dev_lengths,
    y_sequence_length: y_dev_lengths,
    dropout_keep_prob: 1.0
}

#Now, access the op that you want to run. 
op_loss = graph.get_tensor_by_name('loss:0')

print(sess.run(op_loss, feed_dict))
'''
