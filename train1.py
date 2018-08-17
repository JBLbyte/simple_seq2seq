#!/usr/bin/env python
# coding:utf-8


import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from utils import Utils


### hyper parameters
lr = 0.001
epochs = 1000
batch_size = 32
rnn_size = 50
num_layers = 2
encoding_embedding_size = 15
decoding_embedding_size = 15

### data preprocessing
utils = Utils()
x, y = utils.generate_char_x_y(data_size=10000, seed=314)
data = x + y
id2word_x, word2id_x = utils.extract_character_vocab(x)
id2word_y, word2id_y = utils.extract_character_vocab(y)

# 将每一行转换成字符id的list
x_ids = [[word2id_x.get(word, word2id_x['<UNK>'])
    for word in sentence] for sentence in x]
y_ids = [[word2id_y.get(word, word2id_y['<UNK>'])
    for word in sentence] + [word2id_y['<EOS>']] for sentence in y]
print(len(x_ids))
print(len(y_ids))



# 输入层
def get_inputs():
    input_tensor = tf.placeholder(tf.int32, [None,None], name='input_tensor')
    output_tensor = tf.placeholder(tf.int32, [None,None], name='output_tensor')

    # 定义output序列最大长度（之后output_sequence_length和input_sequence_length会作为feed_dict的参数）
    input_sequence_length = tf.placeholder(tf.int32, (None,), name='input_sequence_length')
    output_sequence_length = tf.placeholder(tf.int32, (None,), name='output_sequence_length')
    max_output_sequence_length = tf.reduce_max(output_sequence_length, name='max_output_length')

    return input_tensor, output_tensor, output_sequence_length, max_output_sequence_length, input_sequence_length

def get_lstm_cell(rnn_size):
    lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=314))
    return lstm_cell

# Encoder
def get_encoder_layer(input_tensor,
                      rnn_size,
                      num_layers,
                      input_sequence_length,
                      input_vocab_size,
                      encoding_embedding_size):
    """
    构造Encoder层
    参数说明：
    - input_tensor: 输入tensor
    - rnn_size: rnn隐层结点数量
    - num_layers: 堆叠的rnn cell数量
    - input_sequence_length: 源数据的序列长度
    - input_vocab_size: 源数据的词典大小
    - encoding_embedding_size: embedding的大小
    """
    encoder_embed_input = tf.contrib.layers.embed_sequence(input_tensor, input_vocab_size, encoding_embedding_size)
    cell =  tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])
    encoder_output, encoder_state = tf.nn.dynamic_rnn(
        cell, encoder_embed_input, sequence_length=input_sequence_length, dtype=tf.float32)
    return encoder_output, encoder_state

def process_decoder_input(data, vocab_to_int, batch_size):
    ending = tf.strided_slice(data, [0,0], [batch_size,-1], [1,1])
    decoder_input = tf.concat([tf.fill([batch_size,1], vocab_to_int['<GO>']), ending], 1)
    return decoder_input

# Decoder
def decoding_layer(word2id_y,
                   decoding_embedding_size,
                   num_layers,
                   rnn_size,
                   output_sequence_length,
                   max_output_sequence_length,
                   encoder_state,
                   decoder_input):
    '''
    构造Decoder层
    参数：
    - word2id_y: output数据的映射表
    - decoding_embedding_size: embed向量大小
    - num_layers: 堆叠的RNN单元数量
    - rnn_size: RNN单元的隐层结点数量
    - output_sequence_length: target数据序列长度
    - max_output_sequence_length: target数据序列最大长度
    - encoder_state: encoder端编码的状态向量
    - decoder_input: decoder端输入
    '''

    # Embedding
    output_vocab_size = len(word2id_y)
    decoder_embeddings = tf.Variable(tf.random_uniform([output_vocab_size, decoding_embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

    cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])

    # Output全连接层
    # target_vocab_size定义了输出层的大小
    output_layer = Dense(output_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.1, stddev=0.1))

    # Training decoder
    with tf.variable_scope("decode"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=decoder_embed_input, sequence_length=output_sequence_length, time_major=False)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell, training_helper, encoder_state, output_layer)
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            training_decoder, impute_finished=True, maximum_iterations=max_output_sequence_length)

    # Predicting decoder
    # 与training共享参数
    with tf.variable_scope("decode", reuse=True):
        # 创建一个常量tensor并复制为batch_size的大小
        start_tokens = tf.tile(tf.constant([word2id_y['<GO>']], dtype=tf.int32), [batch_size], name='start_token')
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            decoder_embeddings, start_tokens, word2id_y['<EOS>'])
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                             predicting_helper,
                                                             encoder_state,
                                                             output_layer)
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            predicting_decoder, impute_finished=True, maximum_iterations=max_output_sequence_length)

    return training_decoder_output, predicting_decoder_output

### 构建seq2seq模型，将已经构建完成Encoder和Decoder连接起来
def seq2seq_model(input_tensor,
                  output_tensor,
                  output_sequence_length,
                  max_output_sequence_length,
                  input_sequence_length,
                  input_vocab_size,
                  output_vocab_size,
                  encoder_embedding_size,
                  decoder_embedding_size,
                  rnn_size,
                  num_layers):
    _, encoder_state = get_encoder_layer(input_tensor,
                                         rnn_size,
                                         num_layers,
                                         output_sequence_length,
                                         input_vocab_size,
                                         encoding_embedding_size)
    decoder_input = process_decoder_input(output_tensor, word2id_y, batch_size)
    training_decoder_output, predicting_decoder_output = decoding_layer(word2id_y,
                                                                        decoding_embedding_size,
                                                                        num_layers,
                                                                        rnn_size,
                                                                        output_sequence_length,
                                                                        max_output_sequence_length,
                                                                        encoder_state,
                                                                        decoder_input)
    return training_decoder_output, predicting_decoder_output

def pad_sentence_batch(sentence_batch, pad_id):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length
    参数：
    - sentence batch
    - pad_id: <PAD>对应索引号
    '''
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_id] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def get_batches(inputs, outputs, batch_size, input_pad_int, output_pad_int):
    for batch_i in range(len(inputs)//batch_size):
        start_i = batch_i * batch_size
        inputs_batch = inputs[start_i : start_i + batch_size]
        output_batch = outputs[start_i : start_i + batch_size]

        pad_input_batch = np.array(pad_sentence_batch(inputs_batch, input_pad_int))
        pad_output_batch = np.array(pad_sentence_batch(output_batch, output_pad_int))

        input_lengths = []
        for one_input in inputs_batch:
            input_lengths.append(len(one_input))
        outputs_lengths = []
        for output in output_batch:
            outputs_lengths.append(len(output))
        yield pad_input_batch, pad_output_batch, input_lengths, outputs_lengths


### 构造graph ###
train_graph = tf.Graph()
with train_graph.as_default():
    input_tensor, output_tensor, output_sequence_length, max_output_sequence_length, input_sequence_length = get_inputs()

    training_decoder_output, predicting_decoder_output = seq2seq_model(input_tensor,
                                                                       output_tensor,
                                                                       output_sequence_length,
                                                                       max_output_sequence_length,
                                                                       input_sequence_length,
                                                                       len(word2id_x),
                                                                       len(word2id_y),
                                                                       encoding_embedding_size,
                                                                       decoding_embedding_size,
                                                                       rnn_size,
                                                                       num_layers)

    training_logits = tf.identity(training_decoder_output.rnn_output, name='logits')
    predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')

    #mask是权重的意思
    #tf.sequence_mask([1, 3, 2], 5) # [[True, False, False, False, False],
                                    #  [True, True, True, False, False],
                                    #  [True, True, False, False, False]]
    masks = tf.sequence_mask(output_sequence_length, max_output_sequence_length, dtype=tf.float32, name="masks")

    # logits: A Tensor of shape [batch_size, sequence_length, num_decoder_symbols] and dtype float.
    # The logits correspond to the prediction across all classes at each timestep.
    #output_tensor: A Tensor of shape [batch_size, sequence_length] and dtype int.
    # The target represents the true class at each timestep.
    #weights: A Tensor of shape [batch_size, sequence_length] and dtype float.
    # weights constitutes the weighting of each prediction in the sequence. When using weights as masking,
    # set all valid timesteps to 1 and all padded timesteps to 0, e.g. a mask returned by tf.sequence_mask.
    with tf.name_scope("optimization"):
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            output_tensor,
            masks
        )

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        # minimize函数用于添加操作节点，用于最小化loss，并更新var_list.
        # 该函数是简单的合并了compute_gradients()与apply_gradients()函数返回为一个优化更新后的var_list，
        # 如果global_step非None，该操作还会为global_step做自增操作

        #这里将minimize拆解为了以下两个部分：

        # 对var_list中的变量计算loss的梯度 该函数为函数minimize()的第一部分，返回一个以元组(gradient, variable)组成的列表
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        # 将计算出的梯度应用到变量上，是函数minimize()的第二部分，返回一个应用指定的梯度的操作Operation，对global_step做自增操作
        train_op = optimizer.apply_gradients(capped_gradients)


### Train
x_train = x_ids[:]
y_train = y_ids[:]

display_step = 50

checkpoint = "./log/model/trained_model.ckpt"

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    print()
    for epoch_i in range(1, epochs+1):
        for batch_i, (x_batch, y_batch, x_lengths, y_lengths) in enumerate(get_batches(x_train, y_train, batch_size, word2id_x['<PAD>'], word2id_y['<PAD>'])):
            _, loss = sess.run(
                [train_op, cost],
                feed_dict={
                    input_tensor: x_batch,
                    output_tensor: y_batch,
                    input_sequence_length: x_lengths,
                    output_sequence_length: y_lengths
                })

            if batch_i % display_step == 0:
                print('train loss: {:f}'.format(loss))

                print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}'.format(
                    epoch_i, epochs, batch_i, len(x_train) // batch_size, loss))

    saver = tf.train.Saver()
    saver.save(sess, checkpoint)
    print('Model Trained and Saved')


