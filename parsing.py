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
from lasagne.updates import nesterov_momentum, adam

from neuronlp.io import data_utils
from neuronlp import utils
from neuronlp.layers.recurrent import LSTMLayer
from neuronlp.layers.conv import ConvTimeStep1DLayer
from neuronlp.layers.pool import PoolTimeStep1DLayer
from neuronlp.layers.crf import TreeBiAffineCRFLayer
from neuronlp.objectives import tree_crf_loss
from neuronlp.tasks import parser

WORD_DIM = 100
CHARACTER_DIM = 30


def build_network(word_var, char_var, mask_var, word_alphabet, char_alphabet, num_units, num_types,
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

    # layer_char_input = construct_char_input_layer()
    layer_word_input = construct_word_input_layer()
    mask = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_var, name='mask')

    # Construct Bi-directional LSTM-CNNs-CRF with recurrent dropout.
    # conv_window = 3
    # # shape = [batch, n-step, c_dim, char_length]
    # # construct convolution layer
    # # shape = [batch, n-step, c_filters, output_length]
    # cnn_layer = ConvTimeStep1DLayer(incoming1, num_filters=num_filters, filter_size=conv_window, pad='full',
    #                                 nonlinearity=lasagne.nonlinearities.tanh, name='cnn')
    # # infer the pool size for pooling (pool size should go through all time step of cnn)
    # _, _, _, pool_size = cnn_layer.output_shape
    # # construct max pool layer
    # # shape = [batch, n-step, c_filters, 1]
    # pool_layer = PoolTimeStep1DLayer(cnn_layer, pool_size=pool_size)
    # # reshape: [batch, n-step, c_filters, 1] --> [batch, n-step, c_filters]
    # output_cnn_layer = lasagne.layers.reshape(pool_layer, ([0], [1], [2]))
    #
    # # finally, concatenate the two incoming layers together.
    # # shape = [batch, n-step, c_filter&w_dim]
    # incoming = lasagne.layers.concat([output_cnn_layer, incoming2], axis=2)

    # dropout for incoming
    incoming = lasagne.layers.DropoutLayer(layer_word_input, p=0.15, shared_axes=(1,))

    ingate_forward1 = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                          W_cell=lasagne.init.Uniform(range=0.1))
    outgate_forward1 = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                           W_cell=lasagne.init.Uniform(range=0.1))
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate_forward1 = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                              W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell_forward1 = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                        nonlinearity=nonlinearities.tanh)
    lstm_forward1 = LSTMLayer(incoming, num_units, mask_input=mask, grad_clipping=grad_clipping,
                             nonlinearity=nonlinearities.tanh, peepholes=False,
                             ingate=ingate_forward1, outgate=outgate_forward1,
                             forgetgate=forgetgate_forward1, cell=cell_forward1, p=p, name='forward')
    lstm_forward1 = lasagne.layers.DropoutLayer(lstm_forward1, p=0.33, shared_axes=(1,))

    ingate_forward2 = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                           W_cell=lasagne.init.Uniform(range=0.1))
    outgate_forward2 = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                            W_cell=lasagne.init.Uniform(range=0.1))
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate_forward2 = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                               W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell_forward2 = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                         nonlinearity=nonlinearities.tanh)
    lstm_forward2 = LSTMLayer(lstm_forward1, num_units, mask_input=mask, grad_clipping=grad_clipping,
                              nonlinearity=nonlinearities.tanh, peepholes=False,
                              ingate=ingate_forward2, outgate=outgate_forward2,
                              forgetgate=forgetgate_forward2, cell=cell_forward2, p=p, name='forward')

    # ----------------------------------------------------------------------------------------------------

    ingate_backward1 = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                           W_cell=lasagne.init.Uniform(range=0.1))
    outgate_backward1 = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                            W_cell=lasagne.init.Uniform(range=0.1))
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate_backward1 = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                               W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell_backward1 = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                         nonlinearity=nonlinearities.tanh)
    lstm_backward1 = LSTMLayer(incoming, num_units, mask_input=mask, grad_clipping=grad_clipping,
                              nonlinearity=nonlinearities.tanh, peepholes=False, backwards=True,
                              ingate=ingate_backward1, outgate=outgate_backward1,
                              forgetgate=forgetgate_backward1, cell=cell_backward1, p=p, name='backward')
    lstm_backward1 = lasagne.layers.DropoutLayer(lstm_backward1, p=0.33, shared_axes=(1,))

    ingate_backward2 = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                            W_cell=lasagne.init.Uniform(range=0.1))
    outgate_backward2 = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                             W_cell=lasagne.init.Uniform(range=0.1))
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate_backward2 = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                                W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell_backward2 = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                          nonlinearity=nonlinearities.tanh)
    lstm_backward2 = LSTMLayer(lstm_backward1, num_units, mask_input=mask, grad_clipping=grad_clipping,
                               nonlinearity=nonlinearities.tanh, peepholes=False, backwards=True,
                               ingate=ingate_backward2, outgate=outgate_backward2,
                               forgetgate=forgetgate_backward2, cell=cell_backward2, p=p, name='backward')

    # -----------------------------------------------------------------------------------------------------------

    # concatenate the outputs of forward and backward LSTMs to combine them.
    bi_lstm_cnn = lasagne.layers.concat([lstm_forward2, lstm_backward2], axis=2, name="bi-lstm")
    # shape [batch, n-step, num_units]
    bi_lstm_cnn = lasagne.layers.DropoutLayer(bi_lstm_cnn, p=0.33, shared_axes=(1,))
    # shape [batch, n-step, num_units]
    bi_lstm_cnn = lasagne.layers.DenseLayer(bi_lstm_cnn, 100, nonlinearity=nonlinearities.elu, num_leading_axes=2)

    return TreeBiAffineCRFLayer(bi_lstm_cnn, num_types, mask_input=mask)


def main():
    args_parser = argparse.ArgumentParser(description='Neural MST-Parser')
    args_parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    args_parser.add_argument('--batch_size', type=int, default=10, help='Number of sentences in each batch')
    args_parser.add_argument('--num_units', type=int, default=100, help='Number of hidden units in LSTM')
    args_parser.add_argument('--num_filters', type=int, default=20, help='Number of filters in CNN')
    args_parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    args_parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate of learning rate')
    args_parser.add_argument('--grad_clipping', type=float, default=0, help='Gradient clipping')
    args_parser.add_argument('--gamma', type=float, default=1e-6, help='weight for regularization')
    args_parser.add_argument('--delta', type=float, default=0.0, help='weight for expectation-linear regularization')
    args_parser.add_argument('--regular', choices=['none', 'l2'], help='regularization for training', required=True)
    args_parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    args_parser.add_argument('--schedule', nargs='+', type=int, help='schedule for learning rate decay')
    args_parser.add_argument('--output_prediction', action='store_true', help='Output predictions to temp files')
    args_parser.add_argument('--punctuation', default=None, help='List of punctuations separated by whitespace')
    args_parser.add_argument('--train', help='path of training data')
    args_parser.add_argument('--dev', help='path of validation data')
    args_parser.add_argument('--test', help='path of test data')
    args_parser.add_argument('--tmp', default='tmp', help='Directory for temp files.')

    args = args_parser.parse_args()

    logger = utils.get_logger("Parsing")
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
    punctuation = args.punctuation
    tmp_dir = args.tmp

    punct_set = None
    if punctuation is not None:
        punct_set = set(punctuation.split())
        logger.info("punctuations: %s" % ' '.join(punct_set))

    logger.info("Creating Alphabets")
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet = data_utils.create_alphabets("data/alphabets/",
                                                                                            [train_path, dev_path,
                                                                                             test_path],
                                                                                            40000)
    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())
    logger.info("Type Alphabet Size: %d" % type_alphabet.size())

    num_pos = pos_alphabet.size() - 1
    num_types = type_alphabet.size()

    logger.info("Reading Data")
    data_train = data_utils.read_data(train_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    data_dev = data_utils.read_data(dev_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    data_test = data_utils.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)

    num_data = sum([len(bucket) for bucket in data_train])

    logger.info("constructing network...")
    # create variables
    head_var = T.imatrix(name='heads')
    type_var = T.imatrix(name='types')
    mask_var = T.matrix(name='masks', dtype=theano.config.floatX)
    word_var = T.imatrix(name='inputs')
    char_var = None  # T.itensor3(name='char-inputs')

    network = build_network(word_var, char_var, mask_var, word_alphabet, char_alphabet, num_units, num_types,
                            grad_clipping, num_filters, p=dropout)

    logger.info("Network structure: hidden=%d, filter=%d, dropout=%s" % (num_units, num_filters, dropout))
    # compute loss
    # num_tokens = mask_var.sum(dtype=theano.config.floatX)

    # get outpout of bi-lstm-cnn-crf shape [batch, length, num_labels, num_labels]
    energies_train = lasagne.layers.get_output(network)
    energies_eval = lasagne.layers.get_output(network, deterministic=True)

    # loss_train = tree_crf_loss(energies_train, head_var, type_var, mask_var).mean()
    # loss_eval = tree_crf_loss(energies_eval, head_var, type_var, mask_var).mean()
    loss_train, E, D, L, lengths = tree_crf_loss(energies_train, head_var, type_var, mask_var)
    loss_train = loss_train.mean()
    loss_eval, _, _, _, _ = tree_crf_loss(energies_eval, head_var, type_var, mask_var)
    loss_eval = loss_eval.mean()

    # l2 regularization?
    if regular == 'l2':
        l2_penalty = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
        loss_train = loss_train + gamma * l2_penalty

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = adam(loss_train, params=params, learning_rate=learning_rate, beta1=0.9, beta2=0.9)

    # Compile a function performing a training step on a mini-batch
    train_fn = theano.function([word_var, head_var, type_var, mask_var], [loss_train, E, D, L, lengths], updates=updates)
    # Compile a second function evaluating the loss and accuracy of network
    eval_fn = theano.function([word_var, head_var, type_var, mask_var], [loss_eval, energies_eval])

    # Finally, launch the training loop.
    logger.info(
        "Start training: regularization: %s(%f), dropout: %.2f, delta: %.2f (#training data: %d, batch size: %d, clip: %.1f)..." \
        % (regular, (0.0 if regular == 'none' else gamma), dropout, delta, num_data, batch_size, grad_clipping))

    num_batches = num_data / batch_size + 1
    dev_ucorrect = 0.0
    dev_lcorrect = 0.0
    dev_ucorrect_nopunct = 0.0
    dev_lcorrect_nopunct = 0.0
    best_epoch = 0
    test_ucorrect = 0.0
    test_lcorrect = 0.0
    test_total = 0
    test_total = 0
    test_inst = 0
    lr = learning_rate
    for epoch in range(1, num_epochs + 1):
        print 'Epoch %d (learning rate=%.4f, decay rate=%.4f): ' % (epoch, lr, decay_rate)
        train_err = 0.0
        train_inst = 0
        start_time = time.time()
        num_back = 0
        for batch in xrange(1, num_batches + 1):
            wids, _, pids, hids, tids, masks = data_utils.get_batch(data_train, batch_size)
            err, E, D, L, lengths = train_fn(wids, hids, tids, masks)
            for i in range(wids.shape[0]):
                if lengths[i] < 4:
                    print "\n-------------------------"
                    print D[i, 0:lengths[i], 0:lengths[i]]
                    print E[i, 0:lengths[i], 0:lengths[i]]
                    print L[i, 1:lengths[i], 1:lengths[i]]
                    print '--------------------------'

            train_err += err * wids.shape[0]
            train_inst += wids.shape[0]
            time_ave = (time.time() - start_time) / batch
            time_left = (num_batches - batch) * time_ave

            # update log
            sys.stdout.write("\b" * num_back)
            log_info = 'train: %d/%d loss: %.4f, time left (estimated): %.2fs' % (
                batch, num_batches, train_err / train_inst, time_left)
            sys.stdout.write(log_info)
            num_back = len(log_info)
        # update training log after each epoch
        assert train_inst == num_batches * batch_size
        sys.stdout.write("\b" * num_back)
        print 'train: %d/%d loss: %.4f, time: %.2fs' % (
            train_inst, train_inst, train_err / train_inst, time.time() - start_time)

        # evaluate performance on dev data
        dev_err = 0.0
        dev_ucorr = 0.0
        dev_lcorr = 0.0
        dev_ucorr_nopunc = 0.0
        dev_lcorr_nopunc = 0.0
        dev_total = 0
        dev_total_nopunc = 0
        dev_inst = 0
        for batch in data_utils.iterate_batch(data_dev, batch_size):
            wids, _, pids, hids, tids, masks = batch
            err, energies = eval_fn(wids, hids, tids, masks)
            dev_err += err * wids.shape[0]
            pars_pred, types_pred = parser.decode_MST(energies, masks)
            ucorr, lcorr, total, ucorr_nopunc, \
            lcorr_nopunc, total_nopunc = parser.eval(wids, pids, pars_pred, types_pred, hids, tids, masks,
                                                     tmp_dir + '/dev_parse%d' % epoch, word_alphabet, pos_alphabet,
                                                     type_alphabet, punct_set=punct_set)
            dev_inst += wids.shape[0]

            dev_ucorr += ucorr
            dev_lcorr += lcorr
            dev_total += total

            dev_ucorr_nopunc += ucorr_nopunc
            dev_lcorr_nopunc += lcorr_nopunc
            dev_total_nopunc += total_nopunc
        print 'dev loss: %.4f' % (dev_err / dev_inst)
        print 'W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%' % (
            dev_ucorr, dev_lcorr, dev_total, dev_ucorr * 100 / dev_total, dev_lcorr * 100 / dev_total)
        print 'Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%' % (
            dev_ucorr_nopunc, dev_lcorr_nopunc, dev_total_nopunc, dev_ucorr_nopunc * 100 / dev_total_nopunc,
            dev_lcorr_nopunc * 100 / dev_total_nopunc)

        if dev_ucorrect_nopunct < dev_ucorr_nopunc:
            dev_ucorrect_nopunct = dev_ucorr_nopunc
            dev_lcorrect_nopunct = dev_lcorr_nopunc
            dev_ucorrect = dev_ucorr
            dev_lcorrect = dev_lcorr
            best_epoch = epoch
        print 'best W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%% (epoch: %d)' % (
            dev_ucorrect, dev_lcorrect, dev_total, dev_ucorrect * 100 / dev_total, dev_lcorrect * 100 / dev_total,
            best_epoch)
        print 'best Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%% (epoch: %d)' % (
            dev_ucorrect_nopunct, dev_lcorrect_nopunct, dev_total_nopunc, dev_ucorrect_nopunct * 100 / dev_total_nopunc,
            dev_lcorrect_nopunct * 100 / dev_total_nopunc, best_epoch)

        if epoch in schedule:
            lr = lr * decay_rate
            updates = adam(loss_train, params=params, learning_rate=lr, beta1=0.9, beta2=0.9)
            train_fn = theano.function([word_var, head_var, type_var, mask_var], loss_train, updates=updates)


if __name__ == '__main__':
    main()
