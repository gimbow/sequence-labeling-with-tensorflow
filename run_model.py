#!/usr/bin/env python
# coding=utf8

"""
This is the top-level file to train, evaluate or test your sequence-labeling model
"""

import os
import sys
import time
from collections import namedtuple
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import util
from data import Vocab, ENC
from batcher import Batcher
from model import Model


###############################################################################
FLAGS = tf.app.flags.FLAGS


###############################################################################
# misc path 
tf.app.flags.DEFINE_string("train_data", "./data/*.train", "include wildcards to access multiple datafiles.")
tf.app.flags.DEFINE_string("eval_data", "./data/*.eval", "include wildcards to access multiple datafiles.")
tf.app.flags.DEFINE_string("predict_data", "./data/*.predict", "include wildcards to access multiple datafiles.")
tf.app.flags.DEFINE_string("model_path", "./model/", "Root directory for model.")
tf.app.flags.DEFINE_string("exp_name", "cws", "Root directory for model.")
tf.app.flags.DEFINE_string("log_path", "./log/", "Root directory for all logging.")
tf.app.flags.DEFINE_string("vocab_path", "./data/vocab.txt", "Path expression to text vocabulary file.")

# important settings
tf.app.flags.DEFINE_string("mode", "train", "must be one of train/eval/predict")
tf.app.flags.DEFINE_boolean("single_pass", False, "traverse the corpus for ever or only once.")

# network hyperparameters
tf.app.flags.DEFINE_string("pretrain_embedding", None, "Path express to binary embedding file.")
tf.app.flags.DEFINE_boolean("fine_tune", True, "fine tune the embedding or not.")
tf.app.flags.DEFINE_integer("emb_size", 128, "dimension of word embeddings.")
tf.app.flags.DEFINE_integer("vocab_size", 500000, "Size of vocabulary.")
tf.app.flags.DEFINE_integer("batch_size", 16, "minibatch size")
tf.app.flags.DEFINE_integer("hidden_size", 256, "dimension of RNN hidden states.")
tf.app.flags.DEFINE_integer("max_steps", 400, "max timesteps of sequence.")
tf.app.flags.DEFINE_string("lr_method", "adagrad", "gradient descent method.")
tf.app.flags.DEFINE_float("lr", 0.15, "learning rate.")
tf.app.flags.DEFINE_float("max_grad_norm", 2.0, "clip gradient norm.")
tf.app.flags.DEFINE_boolean("crf", True, "use crf or local softmax to decode.")


###############################################################################
def run_train(model, batcher):
    try:
        while True:
            batch = batcher.next_batch()
            results = model.run_train_step(batch)
            global_step = results['global_step']
            loss = results['loss']
            summary = results['summary']
            if global_step % 100 == 0:
                model.file_writer.add_summary(summary, global_step)
                model.save_session()
    except KeyboardInterrupt:
        tf.logging.info("Caught keyboard interrupt on worker. Close session...")
        model.close_session()
    return 0


def run_eval(model, batcher):
    # load训练好的模型
    model.restore_session()
    
    # 计算准确、召回、F1
    accs = []
    correct_preds, total_correct, total_preds = 0., 0., 0.
    while True:
        batch = batcher.next_batch()
        if batch == None:
            break
        labels_pred, sequence_lengths = model.run_predict_step(batch)
        for lab, lab_pred, length in zip(batch.labels, labels_pred, sequence_lengths):
            lab = lab[:length]
            lab_pred = lab_pred[:length]
            accs += map(lambda element: element[0] == element[1], zip(lab, lab_pred))
            lab_chunks = set(model.vocab.get_chunks(lab))
            lab_pred_chunks = set(model.vocab.get_chunks(lab_pred))
            correct_preds += len(lab_chunks & lab_pred_chunks)
            total_preds += len(lab_pred_chunks)
            total_correct += len(lab_chunks)
    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    acc = np.mean(accs)
    tf.logging.info("- eval acc {:04.2f} - f1 {:04.2f}".format(100 * acc, 100 * f1))
    return acc, f1


def run_predict(model, batcher):
    # load训练好的模型
    model.restore_session()
    while True:
        batch = batcher.next_batch()
        if batch == None:
            break
        labels_pred, sequence_lengths = model.run_predict_step(batch)
        for sentence, length, lab_pred in zip(batch.original_sentences, sequence_lengths, labels_pred):
            sent = sentence.decode(ENC)[:length]
            lab_pred = lab_pred[:length]
            lab_pred_chunks = model.vocab.get_chunks(lab_pred)
            words = [sent[a:b].encode(ENC) for a, b in lab_pred_chunks]
            print sent.encode(ENC)
            print " ".join(words)
        print ""
    return 0


###############################################################################
def main(_):
    # Make a namedtuple hps, containing the all the flag values
    FLAGS.exp_name = FLAGS.model_path + FLAGS.exp_name
    print FLAGS.exp_name
    hps_dict = {}
    for key, val in FLAGS.__flags.iteritems():
        hps_dict[key] = val
    hps = namedtuple("hps", hps_dict.keys())(**hps_dict)
    print hps

    # choose what level of logging you want
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Starting tagging model in %s mode...', (hps.mode))

    # If single_pass=True, check we"re in predict mode
    if hps.mode != "predict" and hps.single_pass:
        raise Exception("The single_pass flag should only be True in predict mode")
        return -1

    # Create a vocabulary
    vocab = Vocab(hps.vocab_path, hps.vocab_size, hps.emb_size)
    # 如果设置了pretain_embedding参数，则预先加载到vocab对象中
    if hps.pretrain_embedding:
        vocab.get_trimmed_embeding(hps.pretrain_embedding)


    # a seed value for randomness
    tf.set_random_seed(111)

    # create a model
    model = Model(vocab, hps)
    model.build_graph()
    if hps.mode == "train":
        # Create a batcher object that will create minibatches of data
        train_batcher = Batcher(hps.train_data, vocab, hps, False)
        run_train(model, train_batcher)
    elif hps.mode == "eval" or hps.mode == "evaluate":
        # Create a batcher object that will create minibatches of data
        eval_batcher = Batcher(hps.eval_data, vocab, hps, True)
        run_eval(model, eval_batcher)
    elif hps.mode == "predict":
        # Create a batcher object that will create minibatches of data
        predict_hps = hps._replace(batch_size=1)
        predict_batcher = Batcher(hps.predict_data, vocab, predict_hps, True)
        run_predict(model, predict_batcher)
    else:
        raise ValueError("The 'mode' flag must be one of train/eval/predict")
        return -1

    ###
    return 0


###############################################################################
if __name__ == "__main__":
    tf.app.run()

