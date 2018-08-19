# coding:utf-8

import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest
from utils import Utils


class Seq2SeqModel(object):

    def __init__(self,
            x_vocab_size,
            y_vocab_size,
            encoder_embedding_size,
            decoder_embedding_size,
            rnn_size,
            num_layers,
            dropout_keep_prob,
            word2id_x,
            word2id_y,
            cell_type='lstm',
            seed=None):
        ### set placeholder
        self.input_x = tf.placeholder(tf.int32, [None,None], name='input_x_tensor')
        self.input_y = tf.placeholder(tf.int32, [None,None], name='input_y_tensor')
        self.x_sequence_length = tf.placeholder(tf.int32, (None,), name='x_sequence_length')
        self.y_sequence_length = tf.placeholder(tf.int32, (None,), name='y_sequence_length')
        self.max_y_sequence_length = tf.reduce_max(self.y_sequence_length, name='max_y_sequence_length')
        self.batch_size = tf.shape(self.input_x)[0]
        masks = tf.sequence_mask(self.y_sequence_length, self.max_y_sequence_length, dtype=tf.float32, name="masks")
        start_tokens = tf.tile(tf.constant([word2id_y['<GO>']], dtype=tf.int32), [self.batch_size], name='start_token')
        end_token = word2id_y['<EOS>']
        
        beam_search = False
        beam_size = 3

        ### encoder
        with tf.name_scope('encoder_embedding'):
            # encoder_embed_input = tf.contrib.layers.embed_sequence(self.input_x, x_vocab_size, encoder_embedding_size)    # high level op, same as following 2 lines
            W_encoder_embedding = tf.Variable(initial_value=tf.random_uniform(shape=[x_vocab_size, encoder_embedding_size], minval=-1.0, maxval=1.0), name='W_encoder_embedding')
            encoder_embed_input = tf.nn.embedding_lookup(W_encoder_embedding, self.input_x)
        
        with tf.name_scope('encoder_rnn'):
            # single direction
            cells_encoder = tf.nn.rnn_cell.MultiRNNCell([self.get_rnn_cell(rnn_size=rnn_size, cell_type=cell_type, dropout_keep_prob=dropout_keep_prob, seed=seed) for _ in range(num_layers)])
            encoder_output, encoder_final_state = tf.nn.dynamic_rnn(cells_encoder, encoder_embed_input, sequence_length=self.x_sequence_length, dtype=tf.float32)

            # bidirection
            # encoder_output, encoder_final_state, _ = self.get_bidirection_rnn_output_and_state(
            #     input_tensor=encoder_embed_input, num_layers=num_layers, rnn_size=rnn_size, cell_type='lstm', dropout_keep_prob=dropout_keep_prob, seed=seed)
        
        ### decoder
        with tf.name_scope('decoder_embedding'):
            ### process decoder input: ['a', 'b', 'c', '<EOS>', '<PAD>'] -> ['<GO>', 'a', 'b', 'c', '<EOS>', '<PAD>']
            input_y_strided = tf.strided_slice(self.input_y, [0,0], [self.batch_size,-1], [1,1], name='input_y_strided')
            decoder_input = tf.concat([tf.fill([self.batch_size,1], word2id_y['<GO>']), input_y_strided], 1, name='input_y_for_decoder')
            W_decoder_embedding = tf.Variable(initial_value=tf.random_uniform(shape=[y_vocab_size, decoder_embedding_size], minval=-1.0, maxval=1.0), name='W_decoder_embedding')
            decoder_embed_input = tf.nn.embedding_lookup(W_decoder_embedding, decoder_input)
        
        with tf.name_scope('decoder_rnn'):
            x_sequence_length = self.x_sequence_length
            if beam_search:
                # 如果使用beam_search，则需要将encoder的输出进行tile_batch，其实就是复制beam_size份。
                encoder_output = tf.contrib.seq2seq.tile_batch(encoder_output, multiplier=beam_size)
                encoder_final_state = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, beam_size), encoder_final_state)
                x_sequence_length = tf.contrib.seq2seq.tile_batch(self.x_sequence_length, multiplier=beam_size)

            # attention mechanism
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(rnn_size, encoder_output, x_sequence_length)
            cells_decoder = tf.nn.rnn_cell.MultiRNNCell([self.get_rnn_cell(rnn_size=rnn_size, cell_type=cell_type, dropout_keep_prob=dropout_keep_prob, seed=seed) for _ in range(num_layers)])
            cells_decoder = tf.contrib.seq2seq.AttentionWrapper(cell=cells_decoder, attention_mechanism=attention_mechanism, attention_layer_size=rnn_size, name='Attention_Wrapper')
            batch_size = self.batch_size * beam_size if beam_search else self.batch_size
            decoder_initial_state = cells_decoder.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=encoder_final_state)
            # decoder_initial_state = encoder_final_state

        with tf.name_scope('output_fc_layer'):
            output_fc_layer = Dense(y_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.1, stddev=0.1, seed=seed), name='output_fc_layer')  # output全连接层，根据y_vocab_size定义输出层的大小

        # training decoder
        with tf.variable_scope("decode_train"):
            train_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input, sequence_length=self.y_sequence_length, time_major=False)
            train_decoder = tf.contrib.seq2seq.BasicDecoder(cells_decoder, train_helper, decoder_initial_state, output_fc_layer)
            train_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder, impute_finished=True, maximum_iterations=self.max_y_sequence_length)
        
        # predicting decoder, 与training共享参数
        with tf.variable_scope('decode_predict'):
            if beam_search:
                predict_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=cells_decoder, embedding=W_decoder_embedding, start_tokens=start_tokens, end_token=end_token, initial_state=decoder_initial_state, beam_width=beam_size, output_layer=output_fc_layer)
            else:
                predict_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(W_decoder_embedding, start_tokens, end_token)
                predict_decoder = tf.contrib.seq2seq.BasicDecoder(cells_decoder, predict_helper, decoder_initial_state, output_fc_layer)
                predict_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predict_decoder, impute_finished=True, maximum_iterations=self.max_y_sequence_length)
            predict_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predict_decoder, maximum_iterations=self.max_y_sequence_length)
        
        with tf.name_scope('output'):
            self.y_logits = tf.identity(train_decoder_output.rnn_output, name='y_logits')
            if beam_search:
                self.y_pred = tf.identity(predict_decoder_output.predicted_ids, name='y_pred')
            else:
                self.y_pred = tf.identity(predict_decoder_output.sample_id, name='y_pred')
            
        with tf.name_scope('loss'):
            self.loss = tf.contrib.seq2seq.sequence_loss(self.y_logits, self.input_y, masks, name='loss')
    

    def get_rnn_cell(self, rnn_size, cell_type=None, dropout_keep_prob=1.0, seed=None):
        if cell_type is None or cell_type == 'lstm':
            rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=seed), name='lstm_cell')
        elif cell_type == 'gru':
            rnn_cell = tf.nn.rnn_cell.GRUCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=seed), name='gru_cell')
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=dropout_keep_prob)
        return rnn_cell
    
    def get_bidirection_rnn_output_and_state(self, input_tensor, num_layers, rnn_size, cell_type=None, dropout_keep_prob=1.0, seed=None):
        cells_fw = [self.get_rnn_cell(rnn_size=rnn_size, cell_type=cell_type, dropout_keep_prob=dropout_keep_prob, seed=seed) for _ in range(num_layers)]
        cells_bw = [self.get_rnn_cell(rnn_size=rnn_size, cell_type=cell_type, dropout_keep_prob=dropout_keep_prob, seed=seed) for _ in range(num_layers)]
        output, state_fw, state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, input_tensor, sequence_length=self.x_sequence_length, dtype=tf.float32)
        return output, state_fw, state_bw
