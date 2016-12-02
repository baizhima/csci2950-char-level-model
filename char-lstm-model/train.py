from adict import adict
from data_loader import load_data, DataLoader
import os
import model
import tensorflow as tf
import numpy as np

flags = tf.flags
flags.DEFINE_string('input_dir', 'data', 'directory to save data')
flags.DEFINE_string('check_dir', 'cv', 'directory of train file')
flags.DEFINE_string('load_saved_checkpoint', None, 'load from checkpoint')
FLAGS = flags.FLAGS

max_epochs = 1


def main(_):

    num_unrolls = 35
    batch_sz = 20
    word_level_set, char_level_set, tensor_words, tensor_chars, max_word_length = load_data(FLAGS.input_dir)
    loader_train = DataLoader(tensor_words['train'], tensor_chars['train'], batch_sz, num_unrolls)
    loader_valid = DataLoader(tensor_words['valid'], tensor_chars['valid'], batch_sz, num_unrolls)
    loader_test = DataLoader(tensor_words['test'], tensor_chars['test'], batch_sz, num_unrolls)

    with tf.Graph().as_default(), tf.Session() as session:
        tf.set_random_seed(1234)
        np.random.seed(seed=1234)

        # TRAIN here
        random_val = 0.04
        initializer = tf.random_uniform_initializer(-random_val, random_val)
        with tf.variable_scope("model", initializer=initializer):
            train_model = model.char_aware_network(
                dropout=0.5,
                batch_sz=batch_sz,
                num_unrolls=num_unrolls,
                char_level_set_size=char_level_set.size(),
                word_level_set_size=word_level_set.size(),
                max_word_length=max_word_length)

            # Loss
            train_model.update(model.cost(train_model.logits, batch_sz, num_unrolls))

            # Train graph
            train_model.update(model.training_graph(train_model.loss * num_unrolls, 1.0))

        saver = tf.train.Saver(max_to_keep=50)

        # VALIDATION here
        with tf.variable_scope("model", reuse=True):
            valid_model = model.char_aware_network(
                batch_sz=batch_sz,
                num_unrolls=num_unrolls,
                char_level_set_size=char_level_set.size(),
                word_level_set_size=word_level_set.size(),
                max_word_length=max_word_length)


            valid_model.update(model.cost(valid_model.logits, batch_sz, num_unrolls))

        if FLAGS.load_saved_checkpoint:
            saver.restore(session, FLAGS.load_saved_checkpoint)
        else:
            tf.initialize_all_variables().run()
            session.run(train_model.scatter_char_embed)

        checkpoint_writer = tf.train.SummaryWriter(FLAGS.check_dir, graph=session.graph)
        lr = 1.0 # need to be tuned
        session.run(tf.assign(train_model.learning_rate, lr))

        state_cur = session.run(train_model.state_init)
        min_loss_for_validation = None
        for epoch in range(max_epochs):

            print("epoch no.", epoch)
            loss_for_training = 0.0
            threshold = 1.e-5 # need to be tuned
            for x, y in loader_train.iter():
                runner = [train_model.loss,train_model.train_step, train_model.state_end, train_model.glob_norm, train_model.glob_step, train_model.scatter_char_embed]
                feed_dict = {train_model.input: x, train_model.target_var: y, train_model.state_init: state_cur}
                loss, _, state_cur, gradient_norm, step, _ = session.run(runner, feed_dict)
                loss_for_training += (loss - loss_for_training) * 0.05

            loss_for_validation = 0.0
            state_cur = session.run(valid_model.state_init)
            for x, y in loader_valid.iter():
                runner = [valid_model.loss, valid_model.state_end]
                feed_dict = {valid_model.input  : x, valid_model.target_var: y, valid_model.state_init: state_cur}
                loss, state_cur = session.run(runner, feed_dict)
                loss_for_validation += loss / loader_valid.length

            print("loss = %f perplexity = %f" % (loss_for_training, np.exp(loss_for_training)))

            if min_loss_for_validation is not None and np.exp(loss_for_validation) > np.exp(min_loss_for_validation) - 1.0:
                cur_lr = session.run(train_model.learning_rate)
                cur_lr *= 0.5
                if cur_lr < threshold:
                    break
                session.run(train_model.learning_rate.assign(cur_lr))
            else:
                min_loss_for_validation = loss_for_validation

            train_loss_val = tf.Summary.Value(tag="train_loss", simple_value=loss_for_training)
            valid_loss_val = tf.Summary.Value(tag="valid_loss", simple_value=loss_for_validation)
            summary_values = [train_loss_val, valid_loss_val]
            summary = tf.Summary(value=summary_values)
            checkpoint_writer.add_summary(summary, step)

            save_dir = '%s/ep_%03d_%.4f.model' % (FLAGS.check_dir, epoch, loss_for_validation)

            saver.save(session, save_dir)
            print(save_dir)


if __name__ == "__main__":
    tf.app.run()
