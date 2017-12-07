#!/usr/bin/env python
# coding=utf8

"""
This file contains code to build and run the tensorflow graph for the sequence-to-sequence model
"""

import os
import time
import numpy as np
import tensorflow as tf
import util


###############################################################################
class Model(object):
    """
    A class to represent a model for sequence labeling.
    """

    def __init__(self, vocab, hps):
        self.vocab = vocab
        self.vocab_size = vocab.get_vocab_size()
        self.emb_size = vocab.get_emb_size()
        self.tag_size = vocab.get_tag_size()
        self.hps = hps
        self.sess = tf.Session(config=util.get_config())
        self.saver = None


    def reinitialize_weights(self, scope_name):
        variables = tf.contrib.framework.get_variables(scope_name)
        init = tf.variables_initializer(variables)
        self.sess.run(init)
        return 0


    def initialize_session(self):
        """
        Defines self.sess and initialize the variables
        """
        tf.logging.info("Initializing tf session")
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.saver = tf.train.Saver()
        return 0


    def restore_session(self):
        """
        Restore weights into session
        """
        hps = self.hps
        tf.logging.info("Restore the latest trained model...")
        self.saver.restore(self.sess, hps.exp_name)
        return 0


    def save_session(self):
        """
        Saves session = weights
        """
        hps = self.hps
        tf.logging.info("Save the latest trained model...")
        if not os.path.exists(hps.model_path):
            os.makedirs(hps.model_path)
        self.saver.save(self.sess, hps.exp_name)
        return 0


    def close_session(self):
        """
        Closes the session
        """
        self.sess.close()
        return 0


    def add_summary_op(self):
        """
        Defines variables for Tensorboard
        Args:
            dir_output: (string) where the results are written
        """
        hps = self.hps
        self.summary     = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(hps.log_path, self.sess.graph)
        return 0


    def add_placeholders(self):
        """
        Add placeholders to the graph. These are entry points for any input data.
        """
        hps = self.hps
        self.lens = tf.placeholder(tf.int32, [None], name="lens")
        self.padding_mask = tf.placeholder(tf.int32, [None, None], name="padding_mask")
        self.unigram_batch = tf.placeholder(tf.int32, [None, None], name="unigram_batch")
        self.bigram_batch = tf.placeholder(tf.int32, [None, None], name="bigram_batch")
        self.labels = tf.placeholder(tf.int32, [None, None], name="labels")
        return 0


    def make_feed_dict(self, batch):
        """
        Make a feed dictionary mapping parts of the batch to the appropriate placeholders.
        """
        feed_dict = {}
        feed_dict[self.lens] = batch.lens
        feed_dict[self.padding_mask] = batch.padding_mask
        feed_dict[self.unigram_batch] = batch.unigram_batch
        feed_dict[self.bigram_batch] = batch.bigram_batch
        feed_dict[self.labels] = batch.labels
        return feed_dict


    def add_model_op(self):
        """
        Add a single-layer bidirectional LSTM to the graph.
        """
        hps = self.hps
        # global step
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        # lookup op
        with tf.variable_scope("embedding_op"):
            # 是否使用pre_train的词向量
            if hps.pretrain_embedding:
                self.embedding = tf.Variable(vocab.embedding, dtype=float32, trainable=hps.fine_tune)
            else:
                self.embedding = tf.get_variable("embedding",
                    shape=[self.vocab_size, self.emb_size], dtype=tf.float32)
            # unigram
            self.unigram_embedding = tf.nn.embedding_lookup(self.embedding,
                    self.unigram_batch, name="unigram_embedding")
            # bigram
            self.bigram_embedding = tf.nn.embedding_lookup(self.embedding,
                    self.bigram_batch, name="bigram_embedding")

        # bilstm op
        with tf.variable_scope("bilstm_op"):
            lstm_cell = tf.contrib.rnn.LSTMCell(hps.hidden_size)
            # unigram
            (unigram_fw, unigram_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell,
                self.unigram_embedding, sequence_length=self.lens, dtype=tf.float32)
            self.unigram_output = tf.concat([unigram_fw, unigram_bw], axis=-1)
            # bigram
            (bigram_fw, bigram_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell,
                self.bigram_embedding, sequence_length=self.lens, dtype=tf.float32)
            self.bigram_output = tf.concat([bigram_fw, bigram_bw], axis=-1)
            self.sum_output = tf.add(self.unigram_output, self.bigram_output)

        # FC op
        with tf.variable_scope("projection_op"):
            W = tf.get_variable("W", shape=[2 * hps.hidden_size, self.tag_size], dtype=tf.float32)
            b = tf.get_variable("b", shape=[self.tag_size], dtype=tf.float32, initializer=tf.zeros_initializer())
            time_steps = tf.shape(self.sum_output)[1]
            self.sum_output = tf.reshape(self.sum_output, [-1, 2 * hps.hidden_size])
            self.pred = tf.matmul(self.sum_output, W) + b
            self.logits = tf.reshape(self.pred, [-1, time_steps, self.tag_size])

        # loss and label op
        with tf.variable_scope("loss_op"):
            if hps.crf:
                log_likelihood, self.trans_params = \
                    tf.contrib.crf.crf_log_likelihood(self.logits, self.labels, self.lens)
                self.loss = tf.reduce_mean(-log_likelihood)
            else:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
                mask = tf.sequence_mask(self.lens)
                losses = tf.boolean_mask(losses, mask)
                self.loss = tf.reduce_mean(losses)
                self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)
            # for tensorboard
            tf.summary.scalar("loss", self.loss)

        # train op
        lr_method = hps.lr_method.lower()
        with tf.variable_scope("train_op"):
            if lr_method == "adam":
                optimizer = tf.train.AdamOptimizer(hps.lr)
            elif lr_method == "adagrad":
                optimizer = tf.train.AdagradOptimizer(hps.lr)
            elif lr_method == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(hps.lr)
            elif lr_method == "rmsprop":
                optimizer = tf.train.RMSPropOptimizer(hps.lr)
            else:
                raise NotImplementedError("Unknown method {}".format(lr_method))
                return -1
            # gradient clipping if clip is positive
            grads, vs     = zip(*optimizer.compute_gradients(self.loss))
            grads, gnorm  = tf.clip_by_global_norm(grads, hps.max_grad_norm)
            # for tensorboard
            tf.summary.scalar("gnorm", gnorm)
            self.train_op = optimizer.apply_gradients(zip(grads, vs), global_step=self.global_step, name="gradients")

        ###
        return 0


    def build_graph(self):
        """
        Add the placeholders, model, summaries to the graph
        """
        self.add_placeholders()
        self.add_model_op()
        self.initialize_session()
        self.add_summary_op()
        return 0


    def run_train_step(self, batch):
        feed_dict = self.make_feed_dict(batch)
        to_return = {
            "global_step": self.global_step,
            "summary": self.summary,
            "loss": self.loss,
            "train_op": self.train_op,
        }
        return self.sess.run(to_return, feed_dict)


    def run_eval_step(self, batch):
        feed_dict = self.make_feed_dict(batch)
        to_return = {
            "global_step": self.global_step,
            "summary": self.summary,
            "loss": self.loss,
        }
        return self.sess.run(to_return, feed_dict)


    def run_predict_step(self, batch):
        """
        viterbi解码
        """
        hps = self.hps
        feed_dict = self.make_feed_dict(batch)
        if hps.crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            sequence_lengths, logits, trans_params = self.sess.run([self.lens, self.logits, self.trans_params], feed_dict)
            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                # keep only the valid steps
                logit = logit[:sequence_length]
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
                viterbi_sequences += [viterbi_seq]
            return viterbi_sequences, sequence_lengths
        else:
            sequence_lengths, labels_pred = self.sess.run([self.lens, self.labels_pred], feed_dict)
            return labels_pred, sequence_lengths


###############################################################################
if __name__ == "__main__":
    pass

