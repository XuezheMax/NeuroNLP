__author__ = 'max'
"""
Implementation of Bi-directional LSTM-CNNs-CRF model for sequence labeling.
"""

import time
import sys
import argparse

import numpy as np
import lasagne
import theano
import theano.tensor as T
from lasagne.layers import Gate
from lasagne import nonlinearities
from lasagne.updates import nesterov_momentum

from neuronlp.io import data_utils, get_logger
from neuronlp import utils
from neuronlp.layers.recurrent import LSTMLayer
from neuronlp.layers.conv import ConvTimeStep1DLayer
from neuronlp.layers.pool import PoolTimeStep1DLayer
from neuronlp.layers.crf import ChainCRFLayer
from neuronlp.objectives import chain_crf_loss, chain_crf_accuracy

WORD_DIM = 100
CHARACTER_DIM = 30


def build_std_dropout(incoming1, incoming2, num_units, num_labels, mask, grad_clipping, num_filters, p):
    # Construct Bi-directional LSTM-CNNs-CRF with standard dropout.
    # first get some necessary dimensions or parameters
    conv_window = 3
    # shape = [batch, n-step, c_dim, char_length]
    incoming1 = lasagne.layers.DropoutLayer(incoming1, p=p)

    # construct convolution layer
    # shape = [batch, n-step, c_filters, output_length]
    cnn_layer = ConvTimeStep1DLayer(incoming1, num_filters=num_filters, filter_size=conv_window, pad='full',
                                    nonlinearity=lasagne.nonlinearities.tanh, name='cnn')
    # infer the pool size for pooling (pool size should go through all time step of cnn)
    _, _, _, pool_size = cnn_layer.output_shape
    # construct max pool layer
    # shape = [batch, n-step, c_filters, 1]
    pool_layer = PoolTimeStep1DLayer(cnn_layer, pool_size=pool_size)
    # reshape: [batch, n-step, c_filters, 1] --> [batch, n-step, c_filters]
    output_cnn_layer = lasagne.layers.reshape(pool_layer, ([0], [1], [2]))

    # finally, concatenate the two incoming layers together.
    # shape = [batch, n-step, c_filter&w_dim]
    incoming = lasagne.layers.concat([output_cnn_layer, incoming2], axis=2)

    # dropout for incoming
    incoming = lasagne.layers.DropoutLayer(incoming, p=p)

    ingate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                          W_cell=lasagne.init.Uniform(range=0.1))
    outgate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                           W_cell=lasagne.init.Uniform(range=0.1))
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                              W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                        nonlinearity=nonlinearities.tanh)
    lstm_forward = lasagne.layers.LSTMLayer(incoming, num_units, mask_input=mask, grad_clipping=grad_clipping,
                                            nonlinearity=nonlinearities.tanh, peepholes=False,
                                            ingate=ingate_forward, outgate=outgate_forward,
                                            forgetgate=forgetgate_forward, cell=cell_forward, name='forward')

    ingate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                           W_cell=lasagne.init.Uniform(range=0.1))
    outgate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                            W_cell=lasagne.init.Uniform(range=0.1))
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                               W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                         nonlinearity=nonlinearities.tanh)
    lstm_backward = lasagne.layers.LSTMLayer(incoming, num_units, mask_input=mask, grad_clipping=grad_clipping,
                                             nonlinearity=nonlinearities.tanh, peepholes=False, backwards=True,
                                             ingate=ingate_backward, outgate=outgate_backward,
                                             forgetgate=forgetgate_backward, cell=cell_backward, name='backward')

    # concatenate the outputs of forward and backward LSTMs to combine them.
    bi_lstm_cnn = lasagne.layers.concat([lstm_forward, lstm_backward], axis=2, name="bi-lstm")

    bi_lstm_cnn = lasagne.layers.DropoutLayer(bi_lstm_cnn, p=p)

    return ChainCRFLayer(bi_lstm_cnn, num_labels, mask_input=mask)


def build_recur_dropout(incoming1, incoming2, num_units, num_labels, mask, grad_clipping, num_filters, p):
    # Construct Bi-directional LSTM-CNNs-CRF with recurrent dropout.
    # first get some necessary dimensions or parameters
    conv_window = 3
    # shape = [batch, n-step, c_dim, char_length]
    # construct convolution layer
    # shape = [batch, n-step, c_filters, output_length]
    cnn_layer = ConvTimeStep1DLayer(incoming1, num_filters=num_filters, filter_size=conv_window, pad='full',
                                    nonlinearity=lasagne.nonlinearities.tanh, name='cnn')
    # infer the pool size for pooling (pool size should go through all time step of cnn)
    _, _, _, pool_size = cnn_layer.output_shape
    # construct max pool layer
    # shape = [batch, n-step, c_filters, 1]
    pool_layer = PoolTimeStep1DLayer(cnn_layer, pool_size=pool_size)
    # reshape: [batch, n-step, c_filters, 1] --> [batch, n-step, c_filters]
    output_cnn_layer = lasagne.layers.reshape(pool_layer, ([0], [1], [2]))

    # finally, concatenate the two incoming layers together.
    # shape = [batch, n-step, c_filter&w_dim]
    incoming = lasagne.layers.concat([output_cnn_layer, incoming2], axis=2)

    # dropout for incoming
    incoming = lasagne.layers.DropoutLayer(incoming, p=p, shared_axes=(1,))

    ingate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                          W_cell=lasagne.init.Uniform(range=0.1))
    outgate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                           W_cell=lasagne.init.Uniform(range=0.1))
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                              W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                        nonlinearity=nonlinearities.tanh)
    lstm_forward = LSTMLayer(incoming, num_units, mask_input=mask, grad_clipping=grad_clipping,
                             nonlinearity=nonlinearities.tanh, peepholes=False,
                             ingate=ingate_forward, outgate=outgate_forward,
                             forgetgate=forgetgate_forward, cell=cell_forward, p=p, name='forward')

    ingate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                           W_cell=lasagne.init.Uniform(range=0.1))
    outgate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                            W_cell=lasagne.init.Uniform(range=0.1))
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                               W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                         nonlinearity=nonlinearities.tanh)
    lstm_backward = LSTMLayer(incoming, num_units, mask_input=mask, grad_clipping=grad_clipping,
                              nonlinearity=nonlinearities.tanh, peepholes=False, backwards=True,
                              ingate=ingate_backward, outgate=outgate_backward,
                              forgetgate=forgetgate_backward, cell=cell_backward, p=p, name='backward')

    # concatenate the outputs of forward and backward LSTMs to combine them.
    bi_lstm_cnn = lasagne.layers.concat([lstm_forward, lstm_backward], axis=2, name="bi-lstm")
    # shape = [batch, n-step, num_units]
    bi_lstm_cnn = lasagne.layers.DropoutLayer(bi_lstm_cnn, p=p, shared_axes=(1,))

    return ChainCRFLayer(bi_lstm_cnn, num_labels, mask_input=mask)


def build_network(word_var, char_var, mask_var, word_alphabet, char_alphabet, dropout, num_units, num_labels,
                  grad_clipping=5.0, num_filters=30, p=0.5):
    def generate_random_embedding(scale, shape):
        return np.random.uniform(-scale, scale, shape).astype(theano.config.floatX)

    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / WORD_DIM)
        table = np.empty([word_alphabet.size(), WORD_DIM], dtype=theano.config.floatX)
        table[data_utils.UNK_ID, :] = generate_random_embedding(scale, [1, WORD_DIM])
        for word, index in word_alphabet.iteritems():
            ww = word.lower() if caseless else word
            embedding = embedd_dict[ww] if ww in embedd_dict else generate_random_embedding(scale, [1, WORD_DIM])
            table[index, :] = embedding
        return table

    def construct_char_embedding_table():
        scale = np.sqrt(3.0 / CHARACTER_DIM)
        table = generate_random_embedding(scale, [char_alphabet.size(), CHARACTER_DIM])
        return table

    def construct_word_input_layer():
        # shape = [batch, n-step]
        layer_word_input = lasagne.layers.InputLayer(shape=(None, None), input_var=word_var, name='word_input')
        # shape = [batch, n-step, w_dim]
        layer_word_embedding = lasagne.layers.EmbeddingLayer(layer_word_input, input_size=word_alphabet.size(),
                                                             output_size=WORD_DIM, W=word_table, name='word_embedd')
        return layer_word_embedding

    def construct_char_input_layer():
        # shape = [batch, n-step, char_length]
        layer_char_input = lasagne.layers.InputLayer(shape=(None, None, data_utils.MAX_CHAR_LENGTH), input_var=char_var,
                                                     name='char_input')

        # shape = [batch, n-step, char_length, c_dim]
        layer_char_embedding = lasagne.layers.EmbeddingLayer(layer_char_input, input_size=char_alphabet.size(),
                                                             output_size=CHARACTER_DIM, W=char_table,
                                                             name='char_embedd')
        # shape = [batch, n-step, c_dim, char_length]
        layer_char_embedding = lasagne.layers.DimshuffleLayer(layer_char_embedding, pattern=(0, 1, 3, 2))
        return layer_char_embedding

    embedd_dict, embedd_dim, caseless = utils.load_word_embedding_dict('glove', "data/glove/glove.6B/glove.6B.100d.gz")
    assert embedd_dim == WORD_DIM

    word_table = construct_word_embedding_table()
    char_table = construct_char_embedding_table()

    layer_char_input = construct_char_input_layer()
    layer_word_input = construct_word_input_layer()
    layer_mask = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_var, name='mask')

    if dropout == 'std':
        return build_std_dropout(layer_char_input, layer_word_input, num_units, num_labels, layer_mask,
                                 grad_clipping, num_filters, p)
    elif dropout == 'recurrent':
        return build_recur_dropout(layer_char_input, layer_word_input, num_units, num_labels, layer_mask,
                                   grad_clipping, num_filters, p)
    else:
        raise ValueError('unkown dropout patten: %s' % dropout)


def main():
    parser = argparse.ArgumentParser(description='Tuning with bi-directional LSTM-CNN-CRF')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of sentences in each batch')
    parser.add_argument('--num_units', type=int, default=100, help='Number of hidden units in LSTM')
    parser.add_argument('--num_filters', type=int, default=20, help='Number of filters in CNN')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate of learning rate')
    parser.add_argument('--grad_clipping', type=float, default=0, help='Gradient clipping')
    parser.add_argument('--gamma', type=float, default=1e-6, help='weight for regularization')
    parser.add_argument('--delta', type=float, default=0.0, help='weight for expectation-linear regularization')
    parser.add_argument('--regular', choices=['none', 'l2'], help='regularization for training', required=True)
    parser.add_argument('--dropout', choices=['std', 'recurrent'], help='dropout patten')
    parser.add_argument('--schedule', nargs='+', type=int, help='schedule for learning rate decay')
    parser.add_argument('--output_prediction', action='store_true', help='Output predictions to temp files')
    parser.add_argument('--train')  # "data/POS-penn/wsj/split1/wsj1.train.original"
    parser.add_argument('--dev')  # "data/POS-penn/wsj/split1/wsj1.dev.original"
    parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"

    args = parser.parse_args()

    logger = get_logger("Sequence Labeling")
    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    num_units = args.num_units
    num_filters = args.num_filters
    regular = args.regular
    grad_clipping = args.grad_clipping
    gamma = args.gamma
    delta = args.delta
    learning_rate = args.learning_rate
    momentum = 0.9
    decay_rate = args.decay_rate
    schedule = args.schedule
    output_predict = args.output_prediction
    dropout = args.dropout
    p = 0.5

    logger.info("Creating Alphabets")
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet = data_utils.create_alphabets("data/alphabets/",
                                                                                            [train_path, dev_path,
                                                                                             test_path],
                                                                                            40000)
    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())

    num_labels = pos_alphabet.size() - 1

    logger.info("Reading Data")
    data_train = data_utils.read_data(train_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    data_dev = data_utils.read_data(dev_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    data_test = data_utils.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)

    num_data = sum([len(bucket) for bucket in data_train])

    logger.info("constructing network...")
    # create variables
    target_var = T.imatrix(name='targets')
    mask_var = T.matrix(name='masks', dtype=theano.config.floatX)
    mask_nr_var = T.matrix(name='masks_nr', dtype=theano.config.floatX)
    word_var = T.imatrix(name='inputs')
    char_var = T.itensor3(name='char-inputs')

    network = build_network(word_var, char_var, mask_var, word_alphabet, char_alphabet, dropout, num_units, num_labels,
                            grad_clipping, num_filters, p)

    logger.info("Network structure: hidden=%d, filter=%d, dropout=%s" % (num_units, num_filters, dropout))
    # compute loss
    num_tokens = mask_var.sum(dtype=theano.config.floatX)
    num_tokens_nr = mask_nr_var.sum(dtype=theano.config.floatX)

    # get outpout of bi-lstm-cnn-crf shape [batch, length, num_labels, num_labels]
    energies_train = lasagne.layers.get_output(network)
    energies_train_det = lasagne.layers.get_output(network, deterministic=True)
    energies_eval = lasagne.layers.get_output(network, deterministic=True)

    loss_train_org = chain_crf_loss(energies_train, target_var, mask_var).mean()

    energy_shape = energies_train.shape
    # [batch, length, num_labels, num_labels] --> [batch*length, num_labels*num_labels]
    energies = T.reshape(energies_train, (energy_shape[0] * energy_shape[1], energy_shape[2] * energy_shape[3]))
    energies = nonlinearities.softmax(energies)
    energies_det = T.reshape(energies_train_det, (energy_shape[0] * energy_shape[1], energy_shape[2] * energy_shape[3]))
    energies_det = nonlinearities.softmax(energies_det)
    # [batch*length, num_labels*num_labels] --> [batch, length*num_labels*num_labels]
    energies = T.reshape(energies, (energy_shape[0], energy_shape[1] * energy_shape[2] * energy_shape[3]))
    energies_det = T.reshape(energies_det, (energy_shape[0], energy_shape[1] * energy_shape[2] * energy_shape[3]))

    loss_train_expect_linear = lasagne.objectives.squared_error(energies, energies_det)
    loss_train_expect_linear = loss_train_expect_linear.sum(axis=1)
    loss_train_expect_linear = loss_train_expect_linear.mean()

    loss_train = loss_train_org + delta * loss_train_expect_linear
    # l2 regularization?
    if regular == 'l2':
        l2_penalty = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
        loss_train = loss_train + gamma * l2_penalty

    _, corr_train = chain_crf_accuracy(energies_train, target_var)
    corr_nr_train = (corr_train * mask_nr_var).sum(dtype=theano.config.floatX)
    corr_train = (corr_train * mask_var).sum(dtype=theano.config.floatX)
    prediction_eval, corr_eval = chain_crf_accuracy(energies_eval, target_var)
    corr_nr_eval = (corr_eval * mask_nr_var).sum(dtype=theano.config.floatX)
    corr_eval = (corr_eval * mask_var).sum(dtype=theano.config.floatX)

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = nesterov_momentum(loss_train, params=params, learning_rate=learning_rate, momentum=momentum)

    # Compile a function performing a training step on a mini-batch
    train_fn = theano.function([word_var, char_var, target_var, mask_var, mask_nr_var],
                               [loss_train, loss_train_org, loss_train_expect_linear,
                                corr_train, corr_nr_train, num_tokens, num_tokens_nr], updates=updates)
    # Compile a second function evaluating the loss and accuracy of network
    eval_fn = theano.function([word_var, char_var, target_var, mask_var, mask_nr_var],
                              [corr_eval, corr_nr_eval, num_tokens, num_tokens_nr, prediction_eval])

    # Finally, launch the training loop.
    logger.info(
        "Start training: regularization: %s(%f), dropout: %s, delta: %.2f (#training data: %d, batch size: %d, clip: %.1f)..." \
        % (regular, (0.0 if regular == 'none' else gamma), dropout, delta, num_data, batch_size, grad_clipping))

    num_batches = num_data / batch_size + 1
    dev_correct = 0.0
    dev_correct_nr = 0.0
    best_epoch = 0
    test_correct = 0.0
    test_correct_nr = 0.0
    test_total = 0
    test_total_nr = 0
    test_inst = 0
    lr = learning_rate
    for epoch in range(1, num_epochs + 1):
        print 'Epoch %d (learning rate=%.4f, decay rate=%.4f): ' % (epoch, lr, decay_rate)
        train_err = 0.0
        train_err_org = 0.0
        train_err_linear = 0.0
        train_corr = 0.0
        train_corr_nr = 0.0
        train_total = 0
        train_total_nr = 0
        train_inst = 0
        start_time = time.time()
        num_back = 0
        for batch in xrange(1, num_batches + 1):
            wids, cids, pids, _, _, masks = data_utils.get_batch(data_train, batch_size)
            masks_nr = np.copy(masks)
            masks_nr[:, 0] = 0
            err, err_org, err_linear, corr, corr_nr, num, num_nr = train_fn(wids, cids, pids, masks, masks_nr)
            train_err += err * wids.shape[0]
            train_err_org += err_org * wids.shape[0]
            train_err_linear += err_linear * wids.shape[0]
            train_corr += corr
            train_corr_nr += corr_nr
            train_total += num
            train_total_nr += num_nr
            train_inst += wids.shape[0]
            time_ave = (time.time() - start_time) / batch
            time_left = (num_batches - batch) * time_ave

            # update log
            sys.stdout.write("\b" * num_back)
            log_info = 'train: %d/%d loss: %.4f, loss_org: %.4f, loss_linear: %.4f, acc: %.2f%%, acc(no root): %.2f%%, time left (estimated): %.2fs' % (
                batch, num_batches, train_err / train_inst, train_err_org / train_inst, train_err_linear / train_inst,
                train_corr * 100 / train_total, train_corr_nr * 100 / train_total_nr, time_left)
            sys.stdout.write(log_info)
            num_back = len(log_info)
        # update training log after each epoch
        assert train_inst == num_batches * batch_size
        assert train_total == train_total_nr + train_inst
        sys.stdout.write("\b" * num_back)
        print 'train: %d/%d loss: %.4f,  loss_org: %.4f, loss_linear: %.4f, acc: %.2f%%, acc(no root): %.2f%%, time: %.2fs' % (
            train_inst, train_inst, train_err / train_inst, train_err_org / train_inst, train_err_linear / train_inst,
            train_corr * 100 / train_total, train_corr_nr * 100 / train_total_nr, time.time() - start_time)

        # evaluate performance on dev data
        dev_corr = 0.0
        dev_corr_nr = 0.0
        dev_total = 0
        dev_total_nr = 0
        dev_inst = 0
        for batch in data_utils.iterate_batch(data_dev, batch_size):
            wids, cids, pids, _, _, masks = batch
            masks_nr = np.copy(masks)
            masks_nr[:, 0] = 0
            corr, corr_nr, num, num_nr, predictions = eval_fn(wids, cids, pids, masks, masks_nr)
            dev_corr += corr
            dev_corr_nr += corr_nr
            dev_total += num
            dev_total_nr += num_nr
            dev_inst += wids.shape[0]
        assert dev_total == dev_total_nr + dev_inst
        print 'dev corr: %d, total: %d, acc: %.2f%%, no root corr: %d, total: %d, acc: %.2f%%' % (
            dev_corr, dev_total, dev_corr * 100 / dev_total, dev_corr_nr, dev_total_nr, dev_corr_nr * 100 / dev_total_nr)

        if dev_correct_nr < dev_corr_nr:
            dev_correct = dev_corr
            dev_correct_nr = dev_corr_nr
            best_epoch = epoch

            # evaluate on test data when better performance detected
            test_corr = 0.0
            test_corr_nr = 0.0
            test_total = 0
            test_total_nr = 0
            test_inst = 0
            for batch in data_utils.iterate_batch(data_test, batch_size):
                wids, cids, pids, _, _, masks = batch
                masks_nr = np.copy(masks)
                masks_nr[:, 0] = 0
                corr, corr_nr, num, num_nr, predictions = eval_fn(wids, cids, pids, masks, masks_nr)
                test_corr += corr
                test_corr_nr += corr_nr
                test_total += num
                test_total_nr += num_nr
                test_inst += wids.shape[0]
            assert test_total + test_total_nr + test_inst
            test_correct = test_corr
            test_correct_nr = test_corr_nr
        print "best dev  corr: %d, total: %d, acc: %.2f%%, no root corr: %d, total: %d, acc: %.2f%% (epoch: %d)" % (
            dev_correct, dev_total, dev_correct * 100 / dev_total,
            dev_correct_nr, dev_total_nr, dev_correct_nr * 100 / dev_total_nr, best_epoch)
        print "best test corr: %d, total: %d, acc: %.2f%%, no root corr: %d, total: %d, acc: %.2f%% (epoch: %d)" % (
            test_correct, test_total, test_correct * 100 / test_total,
            test_correct_nr, test_total_nr, test_correct_nr * 100 / test_total_nr, best_epoch)

        if epoch in schedule:
            lr = lr * decay_rate
            updates = nesterov_momentum(loss_train, params=params, learning_rate=lr, momentum=momentum)
            train_fn = theano.function([word_var, char_var, target_var, mask_var, mask_nr_var],
                                       [loss_train, loss_train_org, loss_train_expect_linear,
                                        corr_train, corr_nr_train, num_tokens, num_tokens_nr], updates=updates)


if __name__ == '__main__':
    main()
