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
            word2id_x,
            word2id_y,
            cell_type='lstm',
            beam_width=3,
            attention_type='Bahdanau', # or 'Luong'
            use_bidirection=True,
            bidirection_layers=1,
            seed=None):
        ### set placeholder
        self.input_x = tf.placeholder(tf.int32, [None,None], name='input_x_tensor')
        self.input_y = tf.placeholder(tf.int32, [None,None], name='input_y_tensor')
        self.x_sequence_length = tf.placeholder(tf.int32, (None,), name='x_sequence_length')
        self.y_sequence_length = tf.placeholder(tf.int32, (None,), name='y_sequence_length')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        self.max_y_sequence_length = tf.reduce_max(self.y_sequence_length, name='max_y_sequence_length')
        self.batch_size = tf.shape(self.input_x)[0]
        masks = tf.sequence_mask(self.y_sequence_length, self.max_y_sequence_length, dtype=tf.float32, name="masks")
        start_tokens = tf.tile(tf.constant([word2id_y['<GO>']], dtype=tf.int32), [self.batch_size], name='start_token')
        end_token = word2id_y['<EOS>']

        ### encoder
        with tf.name_scope('encoder_embedding'):
            # encoder_embed_input = tf.contrib.layers.embed_sequence(self.input_x, x_vocab_size, encoder_embedding_size)    # high level op, same as following 2 lines
            W_encoder_embedding = tf.Variable(initial_value=tf.random_uniform(shape=[x_vocab_size, encoder_embedding_size], minval=-1.0, maxval=1.0), name='W_encoder_embedding')
            encoder_embed_input = tf.nn.embedding_lookup(W_encoder_embedding, self.input_x)

        if not use_bidirection:
            # single direction
            rnn_layers_encoder = [self.get_rnn_cell(rnn_size=rnn_size, cell_type=cell_type, dropout_keep_prob=self.dropout_keep_prob, seed=seed) for _ in range(num_layers)]
            rnn_cells_encoder = tf.nn.rnn_cell.MultiRNNCell(rnn_layers_encoder)
            encoder_output, encoder_final_state = tf.nn.dynamic_rnn(rnn_cells_encoder, encoder_embed_input, sequence_length=self.x_sequence_length, dtype=tf.float32)
        else:
            # bidirection
            encoder_output, _, _ = self.get_bidirection_rnn_output_and_state(
                input_tensor=encoder_embed_input, num_layers=bidirection_layers, rnn_size=rnn_size, cell_type=cell_type, dropout_keep_prob=self.dropout_keep_prob, seed=seed)
            rnn_layers_encoder = [self.get_rnn_cell(rnn_size=rnn_size, cell_type=cell_type, dropout_keep_prob=self.dropout_keep_prob, seed=seed) for _ in range(num_layers)]
            rnn_cells_encoder = tf.nn.rnn_cell.MultiRNNCell(rnn_layers_encoder)
            encoder_output, encoder_final_state = tf.nn.dynamic_rnn(rnn_cells_encoder, encoder_output, sequence_length=self.x_sequence_length, dtype=tf.float32)

        ### decoder
        ### process decoder input: ['a', 'b', 'c', '<EOS>', '<PAD>'] -> ['<GO>', 'a', 'b', 'c', '<EOS>', '<PAD>']
        input_y_strided = tf.strided_slice(self.input_y, [0,0], [self.batch_size,-1], [1,1], name='input_y_strided')
        decoder_input = tf.concat([tf.fill([self.batch_size,1], word2id_y['<GO>']), input_y_strided], 1, name='input_y_for_decoder')

        with tf.name_scope('decoder_embedding'):
            W_decoder_embedding = tf.Variable(initial_value=tf.random_uniform(shape=[y_vocab_size, decoder_embedding_size], minval=-1.0, maxval=1.0), name='W_decoder_embedding')
            decoder_embed_input = tf.nn.embedding_lookup(W_decoder_embedding, decoder_input)

        output_fc_layer = Dense(y_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.1, stddev=0.1, seed=seed), name='output_fc_layer')  # output全连接层，根据y_vocab_size定义输出层的大小

        with tf.variable_scope('decoder_scope'):
            # 使用beam_search，将encoder的输出进行tile_batch，复制beam_width份
            tiled_encoder_output = tf.contrib.seq2seq.tile_batch(encoder_output, 1)
            tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(encoder_final_state, 1)
            y_sequence_length = tf.contrib.seq2seq.tile_batch(self.y_sequence_length, 1)

            rnn_layers_decoder = [self.get_rnn_cell(rnn_size=rnn_size, cell_type=cell_type, dropout_keep_prob=self.dropout_keep_prob, seed=seed) for _ in range(num_layers)]
            rnn_cells_decoder = tf.nn.rnn_cell.MultiRNNCell(rnn_layers_decoder)
            
            with tf.name_scope('attention'):
                attention_cell, decoder_initial_state = self.attention_mechanism(attention_type, rnn_size, rnn_cells_decoder, tiled_encoder_output, tiled_encoder_final_state, self.batch_size, 1)
        
            train_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input, sequence_length=y_sequence_length, time_major=False)
            train_decoder = tf.contrib.seq2seq.BasicDecoder(cell=attention_cell, helper=train_helper, initial_state=decoder_initial_state, output_layer=output_fc_layer)
            train_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=train_decoder, output_time_major=False, impute_finished=True, maximum_iterations=self.max_y_sequence_length)

            self.y_logits = tf.identity(train_decoder_output.rnn_output, name='y_logits')
            self.y_pred = tf.identity(train_decoder_output.sample_id, name='y_pred')

            self.loss = tf.contrib.seq2seq.sequence_loss(self.y_logits, self.input_y, masks, name='loss')

        with tf.variable_scope('decoder_scope', reuse=True):
            tiled_encoder_output_beam = tf.contrib.seq2seq.tile_batch(encoder_output, beam_width)
            tiled_encoder_final_state_beam = tf.contrib.seq2seq.tile_batch(encoder_final_state, beam_width)
            y_sequence_length_beam = tf.contrib.seq2seq.tile_batch(self.y_sequence_length, beam_width)

            rnn_layers_decoder_beam = [self.get_rnn_cell(rnn_size=rnn_size, cell_type=cell_type, dropout_keep_prob=self.dropout_keep_prob, seed=seed) for _ in range(num_layers)]
            rnn_cells_decoder_beam = tf.nn.rnn_cell.MultiRNNCell(rnn_layers_decoder_beam)
            
            with tf.name_scope('attention_beam'):
                attention_cell_beam, decoder_initial_state_beam = self.attention_mechanism(attention_type, rnn_size, rnn_cells_decoder_beam, tiled_encoder_output_beam, tiled_encoder_final_state_beam, self.batch_size, beam_width)

            self.y_pred_beam = tf.identity(self.beam_decoder(attention_cell_beam, W_decoder_embedding, decoder_initial_state_beam, output_fc_layer,
                                                    start_tokens, end_token, beam_width, self.max_y_sequence_length), name='y_pred_beam')
        
        self.training_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    
    def attention_mechanism(self, attention_type, rnn_size, rnn_cells_decoder, encoder_output, encoder_final_state, batch_size, beam_width):
        if attention_type == 'Luong':
            LuongAttention = tf.contrib.seq2seq.LuongAttention(num_units=rnn_size, memory=encoder_output, name='luong_attention')
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(cell=rnn_cells_decoder, attention_mechanism=LuongAttention, attention_layer_size=rnn_size, alignment_history=False)
            initial_state = attention_cell.zero_state(batch_size * beam_width, tf.float32).clone(cell_state=encoder_final_state)
        else:   # attention_type: Bahdanau
            BahdanauAttention = tf.contrib.seq2seq.BahdanauAttention(num_units=rnn_size, memory=encoder_output,	name='Bahdanau')
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(cell=rnn_cells_decoder, attention_mechanism=BahdanauAttention, attention_layer_size=rnn_size, alignment_history=False)
            initial_state = attention_cell.zero_state(batch_size * beam_width, tf.float32).clone(cell_state=encoder_final_state) 
        return attention_cell, initial_state

    def beam_decoder(self, attention_cell, embedding, initial_state, output_layer, start_tokens, end_token, beam_width, max_y_sequence_length):
        decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=attention_cell, embedding=embedding, start_tokens=start_tokens, end_token=end_token, initial_state=initial_state, beam_width=beam_width, output_layer=output_layer)
        decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False, impute_finished=False, maximum_iterations=max_y_sequence_length)	#decoder_output[0].shape = [batch, time, beam]
        trans_output = tf.transpose(decoder_output.predicted_ids, [2,0,1]) # [beam, batch, time]
        best_output = trans_output[0]
        return best_output

    def get_rnn_cell(self, rnn_size, cell_type=None, dropout_keep_prob=1.0, seed=None):
        if cell_type is None or cell_type == 'lstm':
            rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=seed), name='lstm_cell')
        elif cell_type == 'gru':
            rnn_cell = tf.nn.rnn_cell.GRUCell(rnn_size, name='gru_cell')
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=dropout_keep_prob)
        return rnn_cell
    
    def get_bidirection_rnn_output_and_state(self, input_tensor, num_layers, rnn_size, cell_type=None, dropout_keep_prob=1.0, seed=None):
        cells_fw = [self.get_rnn_cell(rnn_size=rnn_size, cell_type=cell_type, dropout_keep_prob=dropout_keep_prob, seed=seed) for _ in range(num_layers)]
        cells_bw = [self.get_rnn_cell(rnn_size=rnn_size, cell_type=cell_type, dropout_keep_prob=dropout_keep_prob, seed=seed) for _ in range(num_layers)]
        outputs, state_fw, state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, input_tensor, sequence_length=self.x_sequence_length, dtype=tf.float32)
        # outputs = tf.transpose(outputs, [1,0,2]) 
        return outputs, state_fw, state_bw
    
    def restore(self, sess, var_list=None, ckpt_path=None):
        if hasattr(self, 'training_variables'):
            var_list = self.training_variables
        self.restorer = tf.train.Saver(var_list)
        self.restorer.restore(sess, ckpt_path)
        print('Restore Finished!')
