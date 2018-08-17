# coding:utf-8

import numpy as np


class Utils(object):

    def __init__(self):
        pass
    
    def generate_char_x_y(self, data_size=10000, seed=None):
        np.random.seed(seed)
        x = []
        y = []
        char_list = np.array([chr(i) for i in range(97, 123)])
        for _ in range(data_size):
            word_len = np.random.randint(3, 10+1, 1)
            rand_char_idx_list = np.random.randint(0, len(char_list), word_len)
            word = char_list[rand_char_idx_list]
            sorted_word = sorted(word)
            x.append(list(word))
            y.append(list(sorted_word))
        return x, y
    
    def extract_character_vocab(self, data):
        data = list(data)
        special_words = ['<PAD>', '<GO>', '<EOS>', '<UNK>']
        word_set = list(set([word for word_list in data for word in word_list]))
        id2word = {idx:word for idx,word in enumerate(special_words + word_set)}
        word2id = {word:idx for idx,word in id2word.items()}
        return id2word, word2id
    
    def pad_sentence_batch(self, sentence_batch, pad_id):
        '''
        对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length
        '''
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + [pad_id] * (max_sentence - len(sentence)) for sentence in sentence_batch]

    def batch_iter(self, num_epochs, xs, ys, batch_size, input_pad_int, output_pad_int, shuffle=True):
        xs = np.array(xs)
        ys = np.array(ys)
        data_size = len(xs)
        num_batches_per_epoch = int((data_size-1)/batch_size)+1
        for epoch in range(num_epochs):
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_xs = xs[shuffle_indices]
                shuffled_ys = ys[shuffle_indices]
            else:
                shuffled_xs = xs
                shuffled_ys = ys
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num*batch_size
                end_index = min((batch_num+1)*batch_size, data_size)
                x_batch = shuffled_xs[start_index:end_index]
                y_batch = shuffled_ys[start_index:end_index]
                pad_x_batch = np.array(self.pad_sentence_batch(x_batch, input_pad_int))
                pad_y_batch = np.array(self.pad_sentence_batch(y_batch, output_pad_int))
                x_lengths = []
                for one_x in x_batch:
                    x_lengths.append(len(one_x))
                y_lengths = []
                for one_y in y_batch:
                    y_lengths.append(len(one_y))
                yield pad_x_batch, pad_y_batch, x_lengths, y_lengths


if __name__ == '__main__':
    utils = Utils()
    x, y = utils.generate_char_x_y(data_size=2, seed=314)
    data = x + y
    id2word_x, word2id_x = utils.extract_character_vocab(x)
    id2word_y, word2id_y = utils.extract_character_vocab(y)

    # 将每一行转换成字符id的list
    x_ids = [[word2id_x.get(word, word2id_x['<UNK>']) for word in sentence] for sentence in x]
    y_ids = [[word2id_y.get(word, word2id_y['<UNK>']) for word in sentence] for sentence in y]

