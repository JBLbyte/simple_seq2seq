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
        self.max_y_sequence_length = tf.reduce_max(self.y_sequence_length, name='max_y_length')
        self.batch_size = tf.shape(self.input_x)[0]

        ### encoder
        encoder_embed_input = tf.contrib.layers.embed_sequence(self.input_x, x_vocab_size, encoder_embedding_size)
        cells_encoder =  tf.contrib.rnn.MultiRNNCell([self.get_lstm_cell(rnn_size, seed) for _ in range(num_layers)])
        encoder_output, encoder_state = tf.nn.dynamic_rnn(
            cells_encoder, encoder_embed_input, sequence_length=self.x_sequence_length, dtype=tf.float32)
        
        ### process decoder input: ['a', 'b', 'c', '<EOS>'] -> ['<GO>', 'a', 'b', 'c']
        ending = tf.strided_slice(self.input_y, [0,0], [self.batch_size,-1], [1,1])
        decoder_input = tf.concat([tf.fill([self.batch_size,1], word2id_y['<GO>']), ending], 1)

        ### decoder
        decoder_embeddings = tf.Variable(tf.random_uniform([y_vocab_size, decoder_embedding_size]))
        decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)
        cells_decoder = tf.contrib.rnn.MultiRNNCell([self.get_lstm_cell(rnn_size) for _ in range(num_layers)])

        # output全连接层，根据y_vocab_size定义输出层的大小
        output_fc_layer = Dense(y_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.1, stddev=0.1, seed=seed))

        # Training decoder
        with tf.variable_scope("decode"):
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=decoder_embed_input, sequence_length=self.y_sequence_length, time_major=False)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(cells_decoder, training_helper, encoder_state, output_fc_layer)
            logits, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                training_decoder, impute_finished=True, maximum_iterations=self.max_y_sequence_length)
        
        y_pred = tf.identity(logits.rnn_output, name='logits')

        masks = tf.sequence_mask(self.y_sequence_length, self.max_y_sequence_length, dtype=tf.float32, name="masks")

        self.loss = tf.contrib.seq2seq.sequence_loss(y_pred, self.input_y, masks)
    

    def get_lstm_cell(self, rnn_size, seed=None):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=seed))
        return lstm_cell
