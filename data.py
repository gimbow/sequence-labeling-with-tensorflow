#!/usr/bin/env python
# coding=utf8

"""
This file contains code to read the train/eval/test data from file and process it, and read the vocab data from file and process it
"""

import sys
import glob
import random
from operator import itemgetter
from collections import OrderedDict
import numpy as np

###############################################################################
# This has a vocab id, which is used to represent OOV words
UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"  # This has a vocab id, which is used to pad the sequence
SENT_START = "<S>"
SENT_END = "</S>"

ENC = "utf8"

# tagset for word segmentation
TAGSET = ["B", "M", "E", "S"]


###############################################################################
def gen_tags(word):
    tag = []
    if len(word) == 1:
        tag = ["S"]
    else:
        tag = ["M"] * len(word)
        tag[0] = "B"
        tag[-1] = "E"
    return tag


def build_vocab(raw_files, vocab_file):
    """
    parse the original file; give the notation and build vocabulary
    """
    fp_vocab = open(vocab_file, "w")
    vocab = OrderedDict()
    for f in raw_files:
        print f
        with open(f, "r") as fp:
            fout = open(f + ".new", "w")
            for line in fp:
                line = line.decode(ENC).strip()
                tokens = line.split()
                # gen tags
                tags = []
                for item in tokens:
                    tags.extend(gen_tags(item))
                # unigram
                sentence = "".join(tokens)
                for i in xrange(len(sentence)):
                    key = sentence[i].encode(ENC)
                    vocab[key] = vocab.get(key, 0) + 1
                # bigram
                key = SENT_START + sentence[:1].encode(ENC)
                vocab[key] = vocab.get(key, 0) + 1
                for i in xrange(1, len(sentence) - 1):
                    key = sentence[i:i + 2].encode(ENC)
                    vocab[key] = vocab.get(key, 0) + 1
                print >> fout, "%s\t%s" % (sentence.encode(ENC), "".join(tags))
            fout.close()
    sorted_dict = sorted(vocab.iteritems(), key=itemgetter(1), reverse=True)
    for key, value in sorted_dict:
        print >> fp_vocab, "%s\t%d" % (key, value)
    fp_vocab.close()
    return 0


###############################################################################
class Vocab(object):
    """Vocabulary class for mapping between words and ids (integers)"""

    def __init__(self, vocab_file, max_size, emb_size):
        """
        Creates a vocab of up to max_size words, reading from the vocab_file.
            If max_size is 0, reads the entire vocab file.
        Args:
          vocab_file: path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, \
            sorted with most frequent word first. This code doesn't actually use the frequencies, though.
          max_size: integer. The maximum size of the resulting Vocabulary.
        """
        self.tag_to_id = dict(zip(TAGSET, xrange(len(TAGSET))))
        self.id_to_tag = {idx: tag for tag, idx in self.tag_to_id.items()}
        self.word_to_id = {}
        self.id_to_word = {}
        self.count = 0  # keeps track of total number of words in the Vocab
        self.emb_size = emb_size

        # [UNK], [PAD], get the ids.
        for w in [UNK_TOKEN, PAD_TOKEN]:
            self.word_to_id[w] = self.count
            self.id_to_word[self.count] = w
            self.count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, "r") as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    print "Warning: incorrectly formatted line in vocabulary file: %s\n" % line
                    continue
                w = pieces[0]
                if w in [UNK_TOKEN, PAD_TOKEN]:
                    raise Exception(
                        "[UNK], [PAD] shouldn\'t be in the vocab file, but %s is" % w)
                if w in self.word_to_id:
                    raise Exception(
                        "Duplicated word in vocabulary file: %s" % w)
                self.word_to_id[w] = self.count
                self.id_to_word[self.count] = w
                self.count += 1
                if max_size != 0 and self.count >= max_size:
                    print "max_size of vocab was specified as %i; Stopping reading." % (max_size, self.count)
                    break
        print "Finished constructing vocabulary of %i total words." % (self.count)

    def word2id(self, word):
        """
        Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV.
        """
        if word not in self.word_to_id:
            return self.word_to_id[UNK_TOKEN]
        return self.word_to_id[word]

    def id2word(self, word_id):
        """
        Returns the word (string) corresponding to an id (integer).
        """
        if word_id not in self.id_to_word:
            raise ValueError("Id not found in vocab: %d" % word_id)
        return self.id_to_word[word_id]

    def tag2id(self, tag):
        return self.tag_to_id[tag]

    def id2tag(self, tag_id):
        return self.id_to_tag[tag_id]

    def get_vocab_size(self):
        return self.count

    def get_tag_size(self):
        return len(self.tag_to_id)

    def get_emb_size(self):
        return self.emb_size

    def get_chunks(self, seq):
        chunks = []
        start = 0
        for index, ch in enumerate(seq):
            if self.id_to_tag[ch] == "E" or self.id_to_tag[ch] == "S":
                chunks.append((start, index + 1))
                start = index + 1
        chunks.append((start, index + 1))
        return chunks


    ###########################################################################
    def set_trimmed_embedding(self, pretrain_file, trimmed_file):
        """
        Saves word embedding embedding in numpy array
        """
        embedding = np.zeros([len(self.word_to_id), self.emb_size])
        try:
            with open(pretrain_file, "r") as fp:
                for line in fp:
                    line = line.strip().split()
                    word = line[0]
                    vector = map(float, line[1:])
                    if len(vector) != self.emb_size:
                        print >> sys.stderr, "pretrain embedding size != emb_size"
                        break
                    if word in self.word_to_id:
                        word_idx = self.word_to_id[word]
                        embedding[word_idx] = np.asarray(vector)
        except IOError:
            return -1
        np.savez_compressed(trimmed_file, embedding=embedding)
        return 0


    def get_trimmed_embedding(self, filename):
        self.embedding = np.load(filename)["embedding"]
        return 0


###############################################################################
def example_generator(data_path, single_pass):
    """
    Generates tf.Examples from data files.
    Args:
    data_path:
        Path to data files. Can include wildcards.
    single_pass:
        Boolean. If True, go through the dataset exactly once. Otherwise, generate random examples indefinitely.
    Yields:
        a string line.
    """
    while True:
        filelist = glob.glob(data_path)  # get the list of datafiles
        if single_pass:
            filelist = sorted(filelist)
        else:
            random.shuffle(filelist)
        for f in filelist:
            with open(f, "r") as fp:
                for line in fp:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    yield line
        if single_pass:
            print "example_generator completed reading all datafiles. No more data."
            break


###############################################################################
if __name__ == "__main__":
    #build_vocab(sys.argv[1:-1], sys.argv[-1])
    vocab_file = "data/vocab.txt"
    max_size = 1000000
    emb_size = 50
    pretrain_file = "./data/pretrain_embedding.txt"
    trimmed_file = "./data/pretrain_embedding.npz"
    vocab = Vocab(vocab_file, max_size, emb_size)
    vocab.set_trimmed_embedding(pretrain_file, trimmed_file)
    vocab.get_trimmed_embedding(trimmed_file)
    print vocab.get_chunks([0,2,0,2,3,1])

