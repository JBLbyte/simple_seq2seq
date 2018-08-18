# coding:utf-8

import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from utils import Utils


class Seq2SeqModel(object):

    def __init__(self,
            x_vocab_size,
            y_vocab_size,
            encoder_embedding_size,
            decoder_embedding_size,
            rnn_size,
            num_layers,
            word2id_x,
            word2id_y,
            seed=None):
        ### set placeholder
        self.input_x = tf.placeholder(tf.int32, [None,None], name='input_x_tensor')
        self.input_y = tf.placeholder(tf.int32, [None,None], name='input_y_tensor')
        self.x_sequence_length = tf.placeholder(tf.int32, (None,), name='x_sequence_length')
        self.y_sequence_length = tf.placeholder(tf.int32, (None,), name='y_sequence_length')
        self.max_y_sequence_length = tf.reduce_max(self.y_sequence_length, name='max_y_sequence_length')
        self.batch_size = tf.shape(self.input_x)[0]

        ### encoder
        with tf.name_scope('encoder_embedding'):
            # encoder_embed_input = tf.contrib.layers.embed_sequence(self.input_x, x_vocab_size, encoder_embedding_size)    # high level op, same as following 2 lines
            W_encoder_embedding = tf.Variable(initial_value=tf.random_uniform(shape=[x_vocab_size, encoder_embedding_size], minval=-1.0, maxval=1.0), name='W_encoder_embedding')
            encoder_embed_input = tf.nn.embedding_lookup(W_encoder_embedding, self.input_x)
        
        with tf.name_scope('encoder_rnn'):
            cells_encoder = tf.nn.rnn_cell.MultiRNNCell([self.get_rnn_cell(rnn_size=rnn_size, cell_type='lstm', seed=seed) for _ in range(num_layers)])
            encoder_output, encoder_final_state = tf.nn.dynamic_rnn(
                cells_encoder, encoder_embed_input, sequence_length=self.x_sequence_length, dtype=tf.float32)
        
        ### decoder
        with tf.name_scope('decoder_embedding'):
            ### process decoder input: ['a', 'b', 'c', '<EOS>'] -> ['<GO>', 'a', 'b', 'c']
            input_y_strided = tf.strided_slice(self.input_y, [0,0], [self.batch_size,-1], [1,1], name='input_y_strided')
            decoder_input = tf.concat([tf.fill([self.batch_size,1], word2id_y['<GO>']), input_y_strided], 1, name='input_y_for_decoder')
            W_decoder_embedding = tf.Variable(initial_value=tf.random_uniform(shape=[y_vocab_size, decoder_embedding_size], minval=-1.0, maxval=1.0), name='W_decoder_embedding')
            decoder_embed_input = tf.nn.embedding_lookup(W_decoder_embedding, decoder_input)
        
        with tf.name_scope('decoder_rnn'):
            cells_decoder = tf.nn.rnn_cell.MultiRNNCell([self.get_rnn_cell(rnn_size=rnn_size, cell_type='lstm', seed=seed) for _ in range(num_layers)])

        with tf.name_scope('output_fc_layer'):
            output_fc_layer = Dense(y_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.1, stddev=0.1, seed=seed), name='output_fc_layer')  # output全连接层，根据y_vocab_size定义输出层的大小

        # training decoder
        with tf.variable_scope("decode_train"):
            train_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input, sequence_length=self.y_sequence_length, time_major=False)
            train_decoder = tf.contrib.seq2seq.BasicDecoder(cells_decoder, train_helper, encoder_final_state, output_fc_layer)
            train_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder, impute_finished=True, maximum_iterations=self.max_y_sequence_length)
        
        # predicting decoder, 与training共享参数
        with tf.variable_scope('decode_predict'):
            # 创建一个常量tensor并复制为batch_size的大小
            start_tokens = tf.tile(tf.constant([word2id_y['<GO>']], dtype=tf.int32), [self.batch_size], name='start_token')
            predict_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(W_decoder_embedding, start_tokens, word2id_y['<EOS>'])
            predict_decoder = tf.contrib.seq2seq.BasicDecoder(cells_decoder, predict_helper, encoder_final_state, output_fc_layer)
            predict_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predict_decoder, impute_finished=True, maximum_iterations=self.max_y_sequence_length)
        
        with tf.name_scope('output'):
            self.y_logits = tf.identity(train_decoder_output.rnn_output, name='y_logits')
            self.y_pred = tf.identity(predict_decoder_output.sample_id, name='y_pred')

        with tf.name_scope('loss'):
            masks = tf.sequence_mask(self.y_sequence_length, self.max_y_sequence_length, dtype=tf.float32, name="masks")
            self.loss = tf.contrib.seq2seq.sequence_loss(self.y_logits, self.input_y, masks)
    

    def get_rnn_cell(self, rnn_size, cell_type=None, seed=None):
        if cell_type is None or cell_type == 'lstm':
            rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=seed), name='lstm_cell')
        elif cell_type == 'gru':
            rnn_cell = tf.nn.rnn_cell.GRUCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=seed), name='gru_cell')
        return rnn_cell
