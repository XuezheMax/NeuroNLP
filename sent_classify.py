__author__ = 'max'
"""
Implementation of RNN models for sentiment analysis
"""
from collections import defaultdict
import random
import time
import sys
import argparse

import numpy as np
import lasagne
import theano
import theano.tensor as T
from lasagne.layers import Gate
from lasagne import nonlinearities
from lasagne.updates import adam

from neuronlp.io import get_logger
from neuronlp import utils
from neuronlp.layers.recurrent import LSTMLayer, GRULayer, SGRULayer

UNK = 0
WORD_DIM = 100

_buckets = [10, 20, 30, 40, 50, 60]


def build_RNN(architec, layer_input, layer_mask, num_units, grad_clipping):
    def build_GRU(reset_input):
        resetgate = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None)

        updategate = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None)

        hiden_update = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                            b=lasagne.init.Constant(0.), nonlinearity=nonlinearities.tanh)

        return GRULayer(layer_input, num_units, mask_input=layer_mask, grad_clipping=grad_clipping,
                        resetgate=resetgate, updategate=updategate, hidden_update=hiden_update,
                        reset_input=reset_input, only_return_final=True, p=0.5, name='GRU')

    def build_LSTM():
        ingate = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                      W_cell=lasagne.init.Uniform(range=0.1))

        outgate = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                       W_cell=lasagne.init.Uniform(range=0.1))
        # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
        forgetgate = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                          W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
        # now use tanh for nonlinear function of cell, need to try pure linear cell
        cell = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                    b=lasagne.init.Constant(0.), nonlinearity=nonlinearities.tanh)

        return LSTMLayer(layer_input, num_units, mask_input=layer_mask, grad_clipping=grad_clipping,
                         ingate=ingate, forgetgate=forgetgate, cell=cell, outgate=outgate,
                         peepholes=False, nonlinearity=nonlinearities.tanh,
                         only_return_final=True, p=0.5, name='LSTM')

    def build_SGRU():
        resetgate_hidden = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                         W_cell=lasagne.init.GlorotUniform())

        resetgate_input = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                                W_cell=lasagne.init.GlorotUniform())

        updategate = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                          W_cell=lasagne.init.GlorotUniform())

        hidden_update = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                            b=lasagne.init.Constant(0.), nonlinearity=nonlinearities.tanh)

        return SGRULayer(layer_input, num_units, mask_input=layer_mask, grad_clipping=grad_clipping,
                         resetgate_input=resetgate_input, resetgate_hidden=resetgate_hidden,
                         updategate=updategate, hidden_update=hidden_update,
                         only_return_final=True, p=0.5, name='SGRU')

    if architec == 'gru0':
        return build_GRU(False)
    elif architec == 'gru1':
        return build_GRU(True)
    elif architec == 'lstm':
        return build_LSTM()
    elif architec == 'sgru':
        return build_SGRU()
    else:
        raise ValueError('unkown architecture: %s' % architec)


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
    tid_inputs = np.empty([batch_size, ], dtype=np.int32)

    masks = np.zeros([batch_size, bucket_length], dtype=np.float32)

    for b in xrange(batch_size):
        words, wids, tid = random.choice(data[bucket_id])

        inst_size = len(wids)
        # word ids
        wid_inputs[b, :inst_size] = wids
        wid_inputs[b, inst_size:] = UNK
        # type id
        tid_inputs[b] = tid
        # masks
        masks[b, :inst_size] = 1.0

    return wid_inputs, tid_inputs, masks


def iterate_batch(data, batch_size, shuffle=False):
    bucket_sizes = [len(data[b]) for b in xrange(len(_buckets))]
    total_size = float(sum(bucket_sizes))
    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle((bucket_indices))

    for bucket_id in bucket_indices:
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size == 0:
            continue

        bucket_length = _buckets[bucket_id]
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int32)
        tid_inputs = np.empty([bucket_size, ], dtype=np.int32)

        masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)

        for i, inst in enumerate(data[bucket_id]):
            words, wids, tid = inst
            inst_size = len(wids)
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = UNK
            # type ids
            tid_inputs[i] = tid
            # masks
            masks[i, :inst_size] = 1.0

        indices = None
        if shuffle:
            indices = np.arange(bucket_size)
            np.random.shuffle(indices)
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield wid_inputs[excerpt], tid_inputs[excerpt], masks[excerpt]


def main():
    parser = argparse.ArgumentParser(description='Tuning with bi-directional MAXRU-CNN')
    parser.add_argument('--architec', choices=['sgru', 'lstm', 'gru0', 'gru1'], help='architecture of rnn', required=True)
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of sentences in each batch')
    parser.add_argument('--num_units', type=int, default=100, help='Number of hidden units in TARU')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate of learning rate')
    parser.add_argument('--grad_clipping', type=float, default=0, help='Gradient clipping')
    parser.add_argument('--schedule', nargs='+', type=int, help='schedule for learning rate decay')
    args = parser.parse_args()

    architec = args.architec
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    num_units = args.num_units
    learning_rate = args.learning_rate
    decay_rate = args.decay_rate
    schedule = args.schedule
    grad_clipping = args.grad_clipping
    logger = get_logger("Sentiment Classification (%s)" % (architec))

    def read_dataset(filename):
        data = [[] for _ in _buckets]
        print 'Reading data from %s' % filename
        counter = 0
        with open(filename, "r") as f:
            for line in f:
                counter += 1
                tag, words = line.lower().strip().split(" ||| ")
                words = words.split(" ")
                wids = [w2i[x] for x in words]
                tag = t2i[tag]
                length = len(words)
                for bucket_id, bucket_size in enumerate(_buckets):
                    if length < bucket_size:
                        data[bucket_id].append([words, wids, tag])
                        break

        print "Total number of data: %d" % counter
        return data

    def generate_random_embedding(scale, shape):
        return np.random.uniform(-scale, scale, shape).astype(theano.config.floatX)

    def construct_word_input_layer():
        # shape = [batch, n-step]
        layer_word_input = lasagne.layers.InputLayer(shape=(None, None), input_var=word_var, name='word_input')
        # shape = [batch, n-step, w_dim]
        layer_word_embedding = lasagne.layers.EmbeddingLayer(layer_word_input, input_size=vocab_size,
                                                             output_size=WORD_DIM, W=word_table, name='word_embedd')
        return layer_word_embedding

    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / WORD_DIM)
        table = np.empty([vocab_size, WORD_DIM], dtype=theano.config.floatX)
        table[UNK, :] = generate_random_embedding(scale, [1, WORD_DIM])
        for word, index in w2i.iteritems():
            if index == 0:
                continue
            ww = word.lower() if caseless else word
            embedding = embedd_dict[ww] if ww in embedd_dict else generate_random_embedding(scale, [1, WORD_DIM])
            table[index, :] = embedding
        return table

    # Functions to read in the corpus
    w2i = defaultdict(lambda: len(w2i))
    t2i = defaultdict(lambda: len(t2i))
    UNK = w2i["<unk>"]

    data_train = read_dataset('data/sst1/train.txt')
    w2i = defaultdict(lambda: UNK, w2i)
    data_dev = read_dataset('data/sst1/dev.txt')
    data_test = read_dataset('data/sst1/test.txt')
    vocab_size = len(w2i)
    num_labels = len(t2i)

    embedd_dict, embedd_dim, caseless = utils.load_word_embedding_dict('glove', "data/glove/glove.6B/glove.6B.100d.gz")
    assert embedd_dim == WORD_DIM

    num_data_train = sum([len(bucket) for bucket in data_train])
    num_data_dev = sum([len(bucket) for bucket in data_dev])
    num_data_test = sum([len(bucket) for bucket in data_test])

    logger.info("constructing network...")
    # create variables
    target_var = T.ivector(name='targets')
    mask_var = T.matrix(name='masks', dtype=theano.config.floatX)
    word_var = T.imatrix(name='inputs')

    word_table = construct_word_embedding_table()
    layer_word_input = construct_word_input_layer()
    layer_mask = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_var, name='mask')

    layer_input = layer_word_input

    layer_input = lasagne.layers.DropoutLayer(layer_input, p=0.2)

    layer_rnn = build_RNN(architec, layer_input, layer_mask, num_units, grad_clipping)
    layer_rnn = lasagne.layers.DropoutLayer(layer_rnn, p=0.5)

    network = lasagne.layers.DenseLayer(layer_rnn, num_units=num_labels, nonlinearity=nonlinearities.softmax,
                                        name='softmax')

    # get output of bi-taru-cnn shape=[batch * max_length, #label]
    prediction_train = lasagne.layers.get_output(network)
    prediction_eval = lasagne.layers.get_output(network, deterministic=True)
    final_prediction = T.argmax(prediction_eval, axis=1)

    loss_train = lasagne.objectives.categorical_crossentropy(prediction_train, target_var).mean()
    loss_eval = lasagne.objectives.categorical_crossentropy(prediction_eval, target_var).mean()

    corr_train = lasagne.objectives.categorical_accuracy(prediction_train, target_var).sum()
    corr_eval = lasagne.objectives.categorical_accuracy(prediction_eval, target_var).sum()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = adam(loss_train, params=params, learning_rate=learning_rate, beta1=0.9, beta2=0.9)

    # Compile a function performing a training step on a mini-batch
    train_fn = theano.function([word_var, target_var, mask_var], [loss_train, corr_train], updates=updates)
    # Compile a second function evaluating the loss and accuracy of network
    eval_fn = theano.function([word_var, target_var, mask_var], [corr_eval, final_prediction])

    # Finally, launch the training loop.
    logger.info("%s: (#data: %d, batch size: %d, clip: %.1f)" % (architec, num_data_train, batch_size, grad_clipping))

    num_batches = num_data_train / batch_size + 1
    dev_correct = 0.0
    best_epoch = 0
    test_correct = 0.0
    test_total = 0
    lr = learning_rate
    for epoch in range(1, num_epochs + 1):
        print 'Epoch %d (%s, learning rate=%.4f, decay rate=%.4f): ' % (epoch, architec, lr, decay_rate)
        train_err = 0.0
        train_corr = 0.0
        train_total = 0
        start_time = time.time()
        num_back = 0
        for batch in xrange(1, num_batches + 1):
            wids, tids, masks = get_batch(data_train, batch_size)
            num = wids.shape[0]
            err, corr = train_fn(wids, tids, masks)
            train_err += err * num
            train_corr += corr
            train_total += num
            time_ave = (time.time() - start_time) / batch
            time_left = (num_batches - batch) * time_ave

            # update log
            sys.stdout.write("\b" * num_back)
            log_info = 'train: %d/%d loss: %.4f, acc: %.2f%%, time left (estimated): %.2fs' % (
                batch, num_batches, train_err / train_total, train_corr * 100 / train_total, time_left)
            sys.stdout.write(log_info)
            num_back = len(log_info)
        # update training log after each epoch
        assert train_total == num_batches * batch_size
        sys.stdout.write("\b" * num_back)
        print 'train: %d loss: %.4f, acc: %.2f%%, time: %.2fs' % (
            train_total, train_err / train_total, train_corr * 100 / train_total, time.time() - start_time)

        # evaluate performance on dev data
        dev_corr = 0.0
        dev_total = 0
        for batch in iterate_batch(data_dev, batch_size):
            wids, tids, masks = batch
            num = wids.shape[0]
            corr, predictions = eval_fn(wids, tids, masks)
            dev_corr += corr
            dev_total += num

        assert dev_total == num_data_dev
        print 'dev corr: %d, total: %d, acc: %.2f%%' % (dev_corr, dev_total, dev_corr * 100 / dev_total)

        if dev_correct <= dev_corr:
            dev_correct = dev_corr
            best_epoch = epoch

            # evaluate on test data when better performance detected
            test_corr = 0.0
            test_total = 0
            for batch in iterate_batch(data_test, batch_size):
                wids, tids, masks = batch
                num = wids.shape[0]
                corr, predictions = eval_fn(wids, tids, masks)
                test_corr += corr
                test_total += num

            assert test_total == num_data_test
            test_correct = test_corr
        print "best dev  corr: %d, total: %d, acc: %.2f%% (epoch: %d)" % (
            dev_correct, dev_total, dev_correct * 100 / dev_total, best_epoch)
        print "best test corr: %d, total: %d, acc: %.2f%%(epoch: %d)" % (
            test_correct, test_total, test_correct * 100 / test_total, best_epoch)

        if epoch in schedule:
            lr = lr * decay_rate
            updates = adam(loss_train, params=params, learning_rate=lr, beta1=0.9, beta2=0.9)
            train_fn = theano.function([word_var, target_var, mask_var], [loss_train, corr_train], updates=updates)


if __name__ == '__main__':
    main()
