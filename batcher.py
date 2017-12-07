#!/usr/bin/env python
# coding=utf8

"""
This file contains code to process data into batches
"""

import time
import glob
import Queue
from collections import namedtuple
from random import shuffle
from threading import Thread
from multiprocessing import Process
import numpy as np
import tensorflow as tf
import data


###############################################################################
class Example(object):
    """
    Class representing a train/val/test example for task.
    """

    def __init__(self, sentence, label, vocab, hps):
        """Initializes the Example
        Args:
          sentence: source texts
          label: source label
          vocab: Vocabulary object
          hps: hyperparameters
        """
        self.enc = data.ENC

        # Process the sentence
        sentence = sentence.decode(self.enc)
        if hps is not None and len(sentence) > hps.max_steps:
            sentence = sentence[:hps.max_steps]
            label = label[:hps.max_steps]
        self.sentence = list(sentence)
        self.label = list(label)
        self.len = len(self.sentence)

    def pad_sequence(self, max_len, pad_token):
        """
        Pad the input sequence with pad_token up to max_len.
        """
        while len(self.sentence) < max_len:
            self.sentence.append(pad_token)
        return 0


###############################################################################
class Batch(object):
    """
    a minibatch of train/val/test examples for sequence labeling.
    """

    def __init__(self, example_list, vocab, hps):
        """
        Turns the example_list into a Batch object.
        """
        self.vocab = vocab
        self.hps = hps
        self.enc = data.ENC
        self.pad_token = data.PAD_TOKEN.decode(self.enc)
        self.init_seq(example_list, hps)
        self.save_example_list(example_list)

    def init_seq(self, example_list, hps):
        """
        Initializes the following
        """
        # Determine the maximum length of the input sequence in this batch
        max_seq_len = max([ex.len for ex in example_list])
        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_sequence(max_seq_len, self.pad_token)

        # Initialize the numpy arrays
        # TODO: 代价敏感的loss，通过设置padding_mask和lens实现
        self.lens = np.zeros((hps.batch_size), dtype=np.int32)
        self.padding_mask = np.zeros((hps.batch_size, max_seq_len), dtype=np.int32)
        for i, ex in enumerate(example_list):
            self.lens[i] = ex.len
            for j in xrange(ex.len):
                self.padding_mask[i][j] = 1

        # parse unigram
        self.unigram_batch = np.zeros((hps.batch_size, max_seq_len), dtype=np.int32)
        for i, ex in enumerate(example_list):
            sentence = ex.sentence
            for j in xrange(len(sentence)):
                w = "".join(sentence[j: j + 1])
                self.unigram_batch[i][j] = self.vocab.word2id(w.encode(self.enc))

        # parse bigram
        self.bigram_batch = np.zeros((hps.batch_size, max_seq_len), dtype=np.int32)
        for i, ex in enumerate(example_list):
            sentence = [self.pad_token] + ex.sentence
            for j in xrange(len(sentence) - 1):
                w = "".join(sentence[j: j + 2])
                self.bigram_batch[i][j] = self.vocab.word2id(w.encode(self.enc))

        # parse labels
        self.labels = np.zeros((hps.batch_size, max_seq_len), dtype=np.int32)
        for i, ex in enumerate(example_list):
            label = ex.label
            for j in xrange(len(label)):
                self.labels[i][j] = self.vocab.tag2id(label[j])
        return 0

    def save_example_list(self, example_list):
        self.original_sentences = ["".join(ex.sentence).encode(self.enc) for ex in example_list]
        self.original_labels = ["".join(ex.label) for ex in example_list]
        return 0


###############################################################################
class Batcher(object):
    """
    A class to generate minibatches of data. Buckets examples together based on length of the sequence.
    """
    # max number of batches the batch_queue can hold
    BATCH_QUEUE_MAX = 100
    DEFAULT_NUM_EXAMPLE_Q_THRERAD = 16
    DEFAULT_NUM_BATCH_Q_THRERAD = 4

    def __init__(self, data_path, vocab, hps, single_pass=False):
        """
        Initialize the batcher. Start threads that process the data into batches.
        Args:
          data_path: tf.Example filepattern.
          vocab: Vocabulary object
          hps: hyperparameters
        """
        self.data_path = data_path
        self.vocab = vocab
        self.hps = hps
        self.single_pass = single_pass 

        # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
        self.batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
        self.example_queue = Queue.Queue(self.BATCH_QUEUE_MAX * self.hps.batch_size)

        filelist = glob.glob(self.data_path)
        if len(filelist) == 0:
            tf.logging.warning("%s is empty", self.data_path)
        # Different settings depending on whether we're in single_pass mode or not
        if self.single_pass:
            # just one thread, so we read through the dataset just once
            self.num_example_q_threads = 1
            self.num_batch_q_threads = 1  # just one thread to batch examples
            # only load one batch's worth of examples before bucketing; this essentially means no bucketing
            self.bucketing_cache_size = 1
            # this will tell us when we're finished reading the dataset
            self.finished_reading = False
        else:
            self.num_example_q_threads = min(len(filelist), self.DEFAULT_NUM_EXAMPLE_Q_THRERAD)
            self.num_batch_q_threads = min(len(filelist), self.DEFAULT_NUM_BATCH_Q_THRERAD)
            # how many batches-worth of examples to load into cache before bucketing
            self.bucketing_cache_size = self.BATCH_QUEUE_MAX

        # Start the threads that load the queues
        self.example_q_threads = []
        for _ in xrange(self.num_example_q_threads):
            #self.example_q_threads.append(Process(target=self.fill_example_queue))
            self.example_q_threads.append(Thread(target=self.fill_example_queue))
            self.example_q_threads[-1].daemon = True
            self.example_q_threads[-1].start()
        self.batch_q_threads = []
        for _ in xrange(self.num_batch_q_threads):
            self.batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self.batch_q_threads[-1].daemon = True
            self.batch_q_threads[-1].start()

        # Start a thread that watches the other threads and restarts them if they're dead
        # We don't want a watcher in single_pass mode because the threads shouldn't run forever
        if not self.single_pass:
            self.watch_thread = Thread(target=self.watch_threads)
            self.watch_thread.daemon = True
            self.watch_thread.start()


    def next_batch(self):
        """
        Return a Batch from the batch queue.
        Returns:
          batch: a Batch object, or None if we're in single_pass mode and we've exhausted the dataset.
        """
        # If the batch queue is empty, print a warning
        if self.batch_queue.qsize() == 0:
            tf.logging.warning("Bucket queue size: %i, Input queue size: %i", \
                self.batch_queue.qsize(), self.example_queue.qsize())
            if self.single_pass and self.finished_reading:
                tf.logging.info("Finished reading dataset in single_pass mode.")
                return None
        # get the next Batch
        batch = self.batch_queue.get()
        return batch

    def fill_example_queue(self):
        """
        Reads data from file and processes into Examples which are then placed into the example queue.
        """
        input_gen = data.example_generator(self.data_path, self.single_pass)
        while True:
            try:
                # read the next example from file. sentence and label are both strings.
                line = input_gen.next()
            except StopIteration:  # if there are no more examples:
                tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
                if self.single_pass:
                    tf.logging.info("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
                    self.finished_reading = True
                    break
                else:
                    raise Exception("single_pass mode is off but the example generator is out of data; error.")
            if len(line) == 0: 
                tf.logging.warning("Found an example with empty sentence text. Skipping it.")
            else:
                # Process into an Example.
                fields = line.split("\t")
                if len(fields) == 2:
                    sentence, label = fields[0], fields[1]
                else:
                    sentence, label = fields[0], ""
                example = Example(sentence, label, self.vocab, self.hps)
                # place the Example in the example queue.
                self.example_queue.put(example)


    def fill_batch_queue(self):
        """
        Takes Examples out of example queue, sorts them by sequence length, processes into Batches and places them in the batch queue.
        In predict mode, makes batches that each contain a single example repeated.
        """
        while True:
            if self.hps.mode != "predict":
                # Get bucketing_cache_size-many batches of Examples into a list, then sort
                inputs = []
                for _ in xrange(self.hps.batch_size * self.bucketing_cache_size):
                    inputs.append(self.example_queue.get())
                # sort by length of encoder sequence
                inputs = sorted(inputs, key=lambda inp: inp.len)

                # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
                batches = []
                for i in xrange(0, len(inputs), self.hps.batch_size):
                    batches.append(inputs[i:i + self.hps.batch_size])
                if not self.single_pass:
                    shuffle(batches)
                for b in batches:  # each b is a list of Example objects
                    self.batch_queue.put(Batch(b, self.vocab, self.hps))
            else:  # predict mode
                b = []
                for _ in xrange(self.hps.batch_size):
                    b.append(self.example_queue.get())
                self.batch_queue.put(Batch(b, self.vocab, self.hps))


    def watch_threads(self):
        """
        Watch example queue and batch queue threads and restart if dead.
        """
        while True:
            time.sleep(60)
            for idx, t in enumerate(self.example_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.logging.error("Found example queue thread dead. Restarting.")
                    new_t = Thread(target=self.fill_example_queue)
                    self.example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
            for idx, t in enumerate(self.batch_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.logging.error("Found batch queue thread dead. Restarting.")
                    new_t = Thread(target=self.fill_batch_queue)
                    self.batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()


###############################################################################
if __name__ == "__main__":
    vocab = data.Vocab("./data/vocab.txt", 0, 50)
    print vocab.tag_to_id
    print vocab.id_to_tag

    hps_dict = {
        "batch_size": 4, 
        "max_steps": 50, 
        "mode": "train", 
        "single_pass": False,
    }
    hps = namedtuple("hps", hps_dict.keys())(**hps_dict)

    example = Example("现代化的战舰上", "BMESMES", vocab, hps)
    print example.sentence
    print example.label
    print example.len

    example2 = Example("现代化的战舰", "BMESME", vocab, hps)
    example3 = Example("现代化的", "BMES", vocab, hps)
    example4 = Example("现代", "BE", vocab, hps)
    example_list = [example, example2, example3, example4]
    batch = Batch(example_list, vocab, hps)
    print batch.lens
    print batch.padding_mask
    print batch.pad_token
    print batch.unigram_batch
    print batch.bigram_batch
    print batch.labels

    data_path = "./data/*.train"
    batcher = Batcher(data_path, vocab, hps, False)
    for i in range(5):
        print ""
        print "batch %d" % (i)
        batch_get = batcher.next_batch()
        print batch_get.lens
        print batch_get.padding_mask
        print batch_get.pad_token
        for s in batch_get.original_sentences:
            print s
        for l in batch_get.original_labels:
            print l
        print batch_get.unigram_batch
        print batch_get.bigram_batch
        print batch_get.labels


