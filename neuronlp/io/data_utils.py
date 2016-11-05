__author__ = 'max'

import re
import random
import numpy as np
from tensorflow.python.platform import gfile
from reader import CoNLLReader
from alphabet import Alphabet
from .. import utils

# Special vocabulary symbols - we always put them at the start.
ROOT = b"_ROOT"
ROOT_POS = b"_ROOT_POS"
ROOT_TYPE = b"_<ROOT>"
PAD = b"_PAD"
PAD_POS = b"_PAD_POS"
PAD_TYPE = b"_<PAD>"
_START_VOCAB = [ROOT, PAD]

ROOT_ID = 1
PAD_ID = 2

# Regular expressions used to normalize digits.
DIGIT_RE = re.compile(br"\d")

_buckets = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]


def create_alphabets(alphabet_directory, data_paths, max_vocabulary_size, normalize_digits=True):
    logger = utils.get_logger("Create Alphabets")
    word_alphabet = Alphabet('word')
    pos_alphabet = Alphabet('pos')
    type_alphabet = Alphabet('type')
    if not gfile.Exists(alphabet_directory):
        logger.info("Creating Alphabets: %s" % alphabet_directory)

        pos_alphabet.add(ROOT_POS)
        type_alphabet.add(ROOT_TYPE)

        pos_alphabet.add(PAD_POS)
        type_alphabet.add(PAD_TYPE)

        vocab = dict()
        for data_path in data_paths:
            logger.info("Processing data: %s" % data_path)
            with gfile.GFile(data_path, mode="r") as file:
                for line in file:
                    line.decode('utf-8')
                    line = line.strip()
                    if len(line) == 0:
                        continue

                    tokens = line.split()
                    word = DIGIT_RE.sub(b"0", tokens[1]) if normalize_digits else tokens[1]
                    pos = tokens[4]
                    type = tokens[7]

                    pos_alphabet.add(pos)
                    type_alphabet.add(type)

                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        logger.info("Total Vocabulary Size: %d" % len(vocab_list))
        logger.info("POS Alphabet Size: %d" % pos_alphabet.size())
        logger.info("Type Alphabet Size: %d" % type_alphabet.size())

        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        for word in vocab_list:
            word_alphabet.add(word)

        word_alphabet.save(alphabet_directory)
        pos_alphabet.save(alphabet_directory)
        type_alphabet.save(alphabet_directory)

    else:
        word_alphabet.load(alphabet_directory)
        pos_alphabet.load(alphabet_directory)
        type_alphabet.load(alphabet_directory)

    word_alphabet.close()
    pos_alphabet.close()
    type_alphabet.close()
    return word_alphabet, pos_alphabet, type_alphabet


def read_data(source_path, word_alphabet, pos_alphabet, type_alphabet, max_size=None, normalize_digits=True):
    logger = utils.get_logger("Reading Data")
    data = [[] for _ in _buckets]

    counter = 0
    reader = CoNLLReader(source_path, word_alphabet, pos_alphabet, type_alphabet)
    inst = reader.getNext(normalize_digits)
    while inst is not None and (not max_size or counter < max_size):
        counter += 1
        if counter % 10000 == 0:
            logger.info("reading data: %d" % counter)

        inst_size = inst.length()
        for bucket_id, bucket_size in enumerate(_buckets):
            if inst_size < bucket_size:
                data[bucket_id].append([inst.word_ids, inst.pos_ids, inst.heads, inst.type_ids])
                break

        inst = reader.getNext(normalize_digits)
    reader.close()
    logger.info("Total number of data: %d" % counter)
    return data


def get_batch(data, batch_size):
    bucket_sizes = [len(data[b]) for b in xrange(len(_buckets))]
    total_size = float(sum(bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    buckets_scale = [sum(bucket_sizes[:i + 1]) / total_size for i in xrange(len(bucket_sizes))]

    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number = np.random.random_sample()
    bucket_id = min([i for i in xrange(len(buckets_scale)) if buckets_scale[i] > random_number])

    bucket_length = _buckets[bucket_id]

    wid_inputs = np.empty([batch_size, bucket_length], dtype=np.int32)
    pid_inputs = np.empty([batch_size, bucket_length], dtype=np.int32)
    hid_inputs = np.empty([batch_size, bucket_length], dtype=np.int32)
    tid_inputs = np.empty([batch_size, bucket_length], dtype=np.int32)

    masks = np.zeros([batch_size, bucket_length], dtype=np.float32)

    for b in xrange(batch_size):
        wids, pids, hids, tids = random.choice(data[bucket_id])
        inst_size = len(wids)
        # word ids
        wid_inputs[b, :inst_size] = wids
        wid_inputs[b, inst_size:] = PAD_ID
        # pos ids
        pid_inputs[b, :inst_size] = pids
        pid_inputs[b, inst_size:] = PAD_ID
        # type ids
        tid_inputs[b, :inst_size] = tids
        tid_inputs[b, inst_size:] = PAD_ID
        # heads
        hid_inputs[b, :inst_size] = hids
        hid_inputs[b, inst_size:] = 0
        # masks
        masks[b, :inst_size] = 1.0

    return wid_inputs, pid_inputs, hid_inputs, tid_inputs, masks
