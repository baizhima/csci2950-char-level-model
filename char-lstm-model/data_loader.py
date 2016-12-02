from adict import adict
import os
import codecs
import numpy as np
import collections


class DataLoader:

    def __init__(self, tensor_word, tensor_char, batch_sz, num_unrolls):
        self.batch_sz = batch_sz
        self.num_unrolls = num_unrolls
        length = tensor_word.shape[0]

        max_len = tensor_char.shape[1]

        total_count = batch_sz * num_unrolls
        adjusted_len = (length // total_count) * total_count 
        tensor_word = tensor_word[:adjusted_len]
        tensor_char = tensor_char[:adjusted_len, :]

        input_file_y = np.zeros_like(tensor_word)
        input_file_y[:-1] = tensor_word[1:].copy()
        input_file_y[-1] = tensor_word[0].copy()
        new_shape_x = [batch_sz, -1, num_unrolls, max_len]
        batch_x = np.transpose(tensor_char.reshape(new_shape_x), axes=(1, 0, 2, 3))
        new_shape_y = [batch_sz, -1, num_unrolls]
        batch_y = np.transpose(input_file_y.reshape(new_shape_y), axes=(1, 0, 2))

        self._batch_x = list(batch_x)
        self._batch_y = list(batch_y)
        self.length = len(self._batch_y)

    def iter(self):        
        for bx, by in zip(self._batch_x, self._batch_y):
            yield bx, by

class Vocabulary:

    def __init__(self, tok_map=None, idx_map=None):
        self.tok_idx_map = {}
        self.idx_tok_map = []
        if tok_map != None and idx_map != None:
            self.tok_idx_map = tok_map
            self.idx_tok_map = idx_map

    @classmethod
    def load(cls, fn):
        with open(fn, 'rb') as f:
            tok_map, idx_map = pickle.load(f)
            return cls(tok_map, idx_map)
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.tok_idx_map, self.idx_tok_map), f, pickle.HIGHEST_PROTOCOL)

    def token(self, index):
        return self.idx_tok_map[index]

    def get(self, tok, default=None):
        return self.tok_idx_map.get(tok, default)

    def add(self, tok):
        if tok not in self.tok_idx_map:
            index = len(self.tok_idx_map)
            self.tok_idx_map[tok] = index
            self.idx_tok_map.append(tok)

        return self.tok_idx_map[tok]


    def size(self):
        return len(self.tok_idx_map)

    def __getitem__(self, tok):
        index = self.get(tok)
        if index is None:
            raise KeyError(tok)
        return index



def load_data(input_dir):

    target_files = ['train', 'valid', 'test']

    char_level_set = Vocabulary()
    char_level_set.add(' ')  
    char_level_set.add('{')  
    char_level_set.add('}')  
    
    word_level_set = Vocabulary()
    word_level_set.add('|')  
    tok_word = collections.defaultdict(list)
    tok_char = collections.defaultdict(list)
    max_len_preset=65

    max_len_from_data = 0
    for fn in target_files:
        fpath = os.path.join(input_dir, fn + '.txt')
        with codecs.open(fpath, 'r', 'utf-8') as f:
            for line in f:
                line = line.strip().replace('}', '').replace('{', '').replace('|', '').replace('+', '')
                for word in line.split():
                    if len(word) > max_len_preset - 2:
                        word = word[:max_len_preset-2]
                    tok_word[fn].append(word_level_set.add(word))
                    char_array = [char_level_set.add(c) for c in '{' + word + '}']
                    tok_char[fn].append(char_array)
                    max_len_from_data = max(max_len_from_data, len(char_array))

                tok_word[fn].append(word_level_set.add('+'))
                char_array = [char_level_set.add(c) for c in '{+}']
                tok_char[fn].append(char_array)


    tensor_words = {}
    tensor_chars = {}
    for fn in target_files:

        tensor_words[fn] = np.array(tok_word[fn], dtype=np.int32)
        tensor_chars[fn] = np.zeros([len(tok_char[fn]), max_len_from_data], dtype=np.int32)

        for i, char_array in enumerate(tok_char[fn]):
            tensor_chars[fn] [i,:len(char_array)] = char_array

    return word_level_set, char_level_set, tensor_words, tensor_chars, max_len_from_data

