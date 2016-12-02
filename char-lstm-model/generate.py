from __future__ import print_function
from adict import adict
import time
import tensorflow as tf
import model
import numpy as np
import os
from data_loader import load_data, DataLoader


flags = tf.flags
flags.DEFINE_string('load_saved_checkpoint', None, 'load model')
flags.DEFINE_string('input_dir', 'data', 'store data')
FLAGS = flags.FLAGS

def main(_):
    word_level_set, char_level_set, tensor_words, tensor_chars, max_word_length = load_data(FLAGS.input_dir)
    with tf.Graph().as_default(), tf.Session() as session:
        tf.set_random_seed(4321)
        np.random.seed(seed=4321)
        with tf.variable_scope("model"):
            m = model.char_aware_network(
                batch_sz=1,
                num_unrolls=1,
                char_level_set_size=char_level_set.size(),
                word_level_set_size=word_level_set.size(),
                max_word_length=max_word_length)

        saver = tf.train.Saver()
        saver.restore(session, FLAGS.load_saved_checkpoint)

        state_cur = session.run(m.state_init)
        logits = np.ones((word_level_set.size(),))
        state_cur = session.run(m.state_init)
        for i in range(300):

            conved_prob = np.exp(logits)
            conved_prob = (conved_prob / np.sum(conved_prob)).ravel()
            ix = np.random.choice(range(len(conved_prob)), p=conved_prob)

            word = word_level_set.token(ix)
            if word == '+':
                print('\n')
            else:
                print(word, end=' ')

            char_data_input = np.zeros((1, 1, max_word_length))
            for i,c in enumerate('{' + word + '}'):
                char_data_input[0,0,i] = char_level_set[c]

            runner = [m.logits, m.state_end]
            feed_dict = {m.input: char_data_input, m.state_init: state_cur}
            logits, state = session.run(runner, feed_dict)
            logits = np.array(logits)

if __name__ == "__main__":
    tf.app.run()
