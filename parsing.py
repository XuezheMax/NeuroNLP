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
from lasagne.updates import nesterov_momentum, adam, total_norm_constraint

from neuronlp.io import data_utils, get_logger
from neuronlp import utils
from neuronlp.layers import get_all_params_by_name
from neuronlp.layers.recurrent import LSTMLayer
from neuronlp.layers.conv import ConvTimeStep1DLayer
from neuronlp.layers.pool import PoolTimeStep1DLayer
from neuronlp.layers.crf import TreeBiAffineCRFLayer
from neuronlp.objectives import tree_crf_loss
from neuronlp.tasks import parser


def build_network(word_var, char_var, pos_var, mask_var, word_alphabet, char_alphabet, pos_alphabet,
                  depth, num_units, num_types, grad_clipping=5.0, num_filters=30, p=0.5, mlp=1, peepholes=False,
                  use_char=False, use_pos=False, normalize_digits=True,
                  embedding='glove', embedding_path='data/glove/glove.6B/glove.6B.100d.gz',
                  char_embedding='random', char_path=None):
    def generate_random_embedding(scale, shape):
        return np.random.uniform(-scale, scale, shape).astype(theano.config.floatX)

    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / WORD_DIM)
        table = np.empty([word_alphabet.size(), WORD_DIM], dtype=theano.config.floatX)
        table[data_utils.UNK_ID, :] = generate_random_embedding(scale, [1, WORD_DIM])
        for word, index in word_alphabet.iteritems():
            ww = word.lower() if caseless else word
            embedd = embedd_dict[ww] if ww in embedd_dict else generate_random_embedding(scale, [1, WORD_DIM])
            table[index, :] = embedd
        print 'construct word table: %s, dimension: %d' % (embedding, table.shape[1])
        return table

    def construct_char_embedding_table():
        if char_embedding == 'random':
            scale = np.sqrt(3.0 / CHARACTER_DIM)
            table = generate_random_embedding(scale, [char_alphabet.size(), CHARACTER_DIM])
        else:
            char_dict, char_dim, caseless = utils.load_word_embedding_dict(char_embedding, char_path,
                                                                           normalize_digits=False)
            scale = np.sqrt(3.0 / char_dim)
            table = np.empty([char_alphabet.size(), char_dim], dtype=theano.config.floatX)
            table[data_utils.UNK_ID, :] = generate_random_embedding(scale, [1, char_dim])
            for char, index in char_alphabet.iteritems():
                cc = char.lower() if caseless else char
                char_embedd = char_dict[cc] if cc in char_dict else generate_random_embedding(scale, [1, char_dim])
                table[index, :] = char_embedd
        print 'construct character table: %s, dimension: %d' % (char_embedding, table.shape[1])
        return table

    def construct_pos_embedding_table():
        scale = np.sqrt(3.0 / POS_DIM)
        table = generate_random_embedding(scale, [pos_alphabet.size(), POS_DIM])
        print 'construct pos table: %s, dimension: %d' % ('random', table.shape[1])
        return table

    def construct_word_input_layer():
        # shape = [batch, n-step]
        layer_word_input = lasagne.layers.InputLayer(shape=(None, None), input_var=word_var, name='word_input')
        # shape = [batch, n-step, w_dim]
        layer_word_embedding = lasagne.layers.EmbeddingLayer(layer_word_input, input_size=word_alphabet.size(),
                                                             output_size=WORD_DIM, W=word_table, name='word_embedd')
        return layer_word_embedding

    def construct_pos_input_layer():
        # shape = [batch, n-step]
        layer_pos_input = lasagne.layers.InputLayer(shape=(None, None), input_var=pos_var, name='pos_input')
        # shape = [batch, n-step, w_dim]
        layer_pos_embedding = lasagne.layers.EmbeddingLayer(layer_pos_input, input_size=pos_alphabet.size(),
                                                            output_size=POS_DIM, W=pos_table, name='pos_embedd')
        return layer_pos_embedding

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

    def construct_bi_lstm_layer():
        lstm_forward = incoming
        lstm_backward = incoming
        assert depth > 0
        for d in xrange(depth):
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
            lstm_forward = LSTMLayer(lstm_forward, num_units, mask_input=mask, grad_clipping=grad_clipping,
                                     nonlinearity=nonlinearities.tanh, peepholes=peepholes,
                                     ingate=ingate_forward, outgate=outgate_forward,
                                     forgetgate=forgetgate_forward, cell=cell_forward, p=p, name='forward%d' % d)
            lstm_forward = lasagne.layers.DropoutLayer(lstm_forward, p=0.33, shared_axes=(1,))
            # ----------------------------------------------------------------------------------------------------
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
            lstm_backward = LSTMLayer(lstm_backward, num_units, mask_input=mask, grad_clipping=grad_clipping,
                                      nonlinearity=nonlinearities.tanh, peepholes=peepholes, backwards=True,
                                      ingate=ingate_backward, outgate=outgate_backward,
                                      forgetgate=forgetgate_backward, cell=cell_backward, p=p, name='backward%d' % d)
            lstm_backward = lasagne.layers.DropoutLayer(lstm_backward, p=0.33, shared_axes=(1,))
            # ------------------------------------------------------------------------------------------------------

        # concatenate the outputs of forward and backward LSTMs to combine them.
        bi_lstm_cnn = lasagne.layers.concat([lstm_forward, lstm_backward], axis=2, name="bi-lstm")
        return bi_lstm_cnn

    embedd_dict, embedd_dim, caseless = utils.load_word_embedding_dict(embedding, embedding_path,
                                                                       normalize_digits=normalize_digits)

    WORD_DIM = embedd_dim
    POS_DIM = 50
    CHARACTER_DIM = 50

    word_table = construct_word_embedding_table()
    pos_table = construct_pos_embedding_table() if use_pos else None
    char_table = construct_char_embedding_table() if use_char else None

    if char_table is not None:
        CHARACTER_DIM = char_table.shape[1]

    layer_word_input = construct_word_input_layer()
    incoming = layer_word_input
    mask = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_var, name='mask')

    if use_pos:
        layer_pos_input = construct_pos_input_layer()
        incoming = lasagne.layers.concat([incoming, layer_pos_input], axis=2)

    if use_char:
        layer_char_input = construct_char_input_layer()
        # dropout before CNN
        # TODO
        # layer_char_input = lasagne.layers.DropoutLayer(layer_char_input, p=0.15)
        # Construct Bi-directional LSTM-CNNs-CRF with recurrent dropout.
        conv_window = 3
        # shape = [batch, n-step, c_dim, char_length]
        # construct convolution layer
        # shape = [batch, n-step, c_filters, output_length]
        cnn_layer = ConvTimeStep1DLayer(layer_char_input, num_filters=num_filters, filter_size=conv_window, pad='full',
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
        incoming = lasagne.layers.concat([output_cnn_layer, incoming], axis=2)

    # dropout for incoming
    incoming = lasagne.layers.DropoutLayer(incoming, p=0.15, shared_axes=(1,))
    # shape [batch, n-step, num_units]
    bi_lstm_cnn = construct_bi_lstm_layer()

    # MLP layers
    # shape [batch, n-step, 100]
    for d in xrange(1, mlp):
        bi_lstm_cnn = lasagne.layers.DenseLayer(bi_lstm_cnn, 100, nonlinearity=nonlinearities.elu,
                                                num_leading_axes=2, name='dense%d' % d)
        bi_lstm_cnn = lasagne.layers.DropoutLayer(bi_lstm_cnn, p=0.33, shared_axes=(1,))

    bi_lstm_cnn = lasagne.layers.DenseLayer(bi_lstm_cnn, 100, nonlinearity=nonlinearities.elu,
                                            num_leading_axes=2, name='dense%d' % mlp)

    return TreeBiAffineCRFLayer(bi_lstm_cnn, num_types, mask_input=mask, name='crf')


def create_updates(loss, network, opt, learning_rate, momentum, beta1, beta2):
    params = lasagne.layers.get_all_params(network, trainable=True)
    grads = theano.grad(loss, params)
    # if max_norm:
    #     names = ['crf.U', 'crf.W_h', 'crf.W_c', 'crf.b']
    #     constraints = [grad for param, grad in zip(params, grads) if param.name in names]
    #     assert len(constraints) == 4
    #     scaled_grads = total_norm_constraint(constraints, max_norm=max_norm)
    #     counter = 0
    #     for i in xrange(len(params)):
    #         param = params[i]
    #         if param.name in names:
    #             grads[i] = scaled_grads[counter]
    #             counter += 1
    #     assert counter == 4
    if opt == 'adam':
        updates = adam(grads, params=params, learning_rate=learning_rate, beta1=beta1, beta2=beta2)
    elif opt == 'momentum':
        updates = nesterov_momentum(grads, params=params, learning_rate=learning_rate, momentum=momentum)
    else:
        raise ValueError('unkown optimization algorithm: %s' % opt)

    return updates


def main():
    args_parser = argparse.ArgumentParser(description='Neural MST-Parser')
    args_parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    args_parser.add_argument('--batch_size', type=int, default=10, help='Number of sentences in each batch')
    args_parser.add_argument('--num_units', type=int, default=100, help='Number of hidden units in LSTM')
    args_parser.add_argument('--depth', type=int, default=2, help='Depth of LSTM layer')
    args_parser.add_argument('--mlp', type=int, default=1, help='Depth of MLP layer')
    args_parser.add_argument('--num_filters', type=int, default=20, help='Number of filters in CNN')
    args_parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    args_parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate of learning rate')
    args_parser.add_argument('--grad_clipping', type=float, default=0, help='Gradient clipping')
    args_parser.add_argument('--peepholes', action='store_true', help='Peepholes for LSTM')
    args_parser.add_argument('--max_norm', type=float, default=0, help='weight for max-norm regularization')
    args_parser.add_argument('--gamma', type=float, default=1e-6, help='weight for regularization')
    args_parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam')
    args_parser.add_argument('--delta', type=float, default=0.0, help='weight for expectation-linear regularization')
    args_parser.add_argument('--regular', choices=['none', 'l2'], help='regularization for training', required=True)
    args_parser.add_argument('--opt', choices=['adam', 'momentum'], help='optimization algorithm', required=True)
    args_parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    args_parser.add_argument('--schedule', nargs='+', type=int, help='schedule for learning rate decay', required=True)
    # args_parser.add_argument('--schedule', type=int, help='schedule for learning rate decay', required=True)
    args_parser.add_argument('--pos', action='store_true', help='using pos embedding')
    args_parser.add_argument('--char', action='store_true', help='using cnn for character embedding')
    args_parser.add_argument('--normalize_digits', action='store_true', help='normalize digits')
    args_parser.add_argument('--output_prediction', action='store_true', help='Output predictions to temp files')
    # args_parser.add_argument('--punctuation', default=None, help='List of punctuations separated by whitespace')
    args_parser.add_argument('--punctuation', nargs='+', type=str, help='List of punctuations')
    args_parser.add_argument('--train', help='path of training data')
    args_parser.add_argument('--dev', help='path of validation data')
    args_parser.add_argument('--test', help='path of test data')
    args_parser.add_argument('--embedding', choices=['glove', 'senna', 'sskip', 'polyglot'], help='Embedding for words',
                             required=True)
    args_parser.add_argument('--char_embedding', choices=['random', 'polyglot'], help='Embedding for characters',
                             required=True)
    args_parser.add_argument('--embedding_dict', default='data/word2vec/GoogleNews-vectors-negative300.bin',
                             help='path for embedding dict')
    args_parser.add_argument('--char_dict', default='data/polyglot/polyglot-zh_char.pkl',
                             help='path for character embedding dict')
    args_parser.add_argument('--tmp', default='tmp', help='Directory for temp files.')

    args = args_parser.parse_args()

    logger = get_logger("Parsing")
    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    num_units = args.num_units
    depth = args.depth
    mlp = args.mlp
    num_filters = args.num_filters
    regular = args.regular
    opt = args.opt
    grad_clipping = args.grad_clipping
    peepholes = args.peepholes
    gamma = args.gamma
    delta = args.delta
    max_norm = args.max_norm
    learning_rate = args.learning_rate
    momentum = 0.9
    beta1 = 0.9
    beta2 = args.beta2
    decay_rate = args.decay_rate
    schedule = args.schedule
    use_pos = args.pos
    use_char = args.char
    normalize_digits = args.normalize_digits
    output_predict = args.output_prediction
    dropout = args.dropout
    punctuation = args.punctuation
    tmp_dir = args.tmp
    embedding = args.embedding
    char_embedding = args.char_embedding
    embedding_path = args.embedding_dict
    char_path = args.char_dict

    punct_set = None
    if punctuation is not None:
        punct_set = set(punctuation)
        logger.info("punctuations(%d): %s" % (len(punct_set), ' '.join(punct_set)))

    logger.info("Creating Alphabets: normalize_digits=%s" % normalize_digits)
    word_alphabet, char_alphabet, \
    pos_alphabet, type_alphabet = data_utils.create_alphabets("data/alphabets/", [train_path,],
                                                              60000, min_occurence=1, normalize_digits=normalize_digits)
    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())
    logger.info("Type Alphabet Size: %d" % type_alphabet.size())

    num_pos = pos_alphabet.size()
    num_types = type_alphabet.size()

    logger.info("Reading Data")
    data_train = data_utils.read_data(train_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                      normalize_digits=normalize_digits)
    data_dev = data_utils.read_data(dev_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                    normalize_digits=normalize_digits)
    data_test = data_utils.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                     normalize_digits=normalize_digits)

    num_data = sum([len(bucket) for bucket in data_train])

    logger.info("constructing network...(pos embedding=%s, character embedding=%s)" % (use_pos, use_char))
    # create variables
    head_var = T.imatrix(name='heads')
    type_var = T.imatrix(name='types')
    mask_var = T.matrix(name='masks', dtype=theano.config.floatX)
    word_var = T.imatrix(name='inputs')
    pos_var = T.imatrix(name='pos-inputs')
    char_var = T.itensor3(name='char-inputs')

    network = build_network(word_var, char_var, pos_var, mask_var, word_alphabet, char_alphabet, pos_alphabet,
                            depth, num_units, num_types, grad_clipping, num_filters,
                            p=dropout, mlp=mlp, peepholes=peepholes,
                            use_char=use_char, use_pos=use_pos, normalize_digits=normalize_digits,
                            embedding=embedding, embedding_path=embedding_path,
                            char_embedding=char_embedding, char_path=char_path)

    logger.info("Network: depth=%d, hidden=%d, peepholes=%s, filter=%d, dropout=%s, #mlp=%d" % (
        depth, num_units, peepholes, num_filters, dropout, mlp))
    # compute loss
    energies_train = lasagne.layers.get_output(network)
    energies_eval = lasagne.layers.get_output(network, deterministic=True)

    loss_train = tree_crf_loss(energies_train, head_var, type_var, mask_var).mean()
    loss_eval = tree_crf_loss(energies_eval, head_var, type_var, mask_var).mean()
    # loss_train, E, D, L, lengths = tree_crf_loss(energies_train, head_var, type_var, mask_var)
    # loss_train = loss_train.mean()
    # loss_eval, _, _, _, _ = tree_crf_loss(energies_eval, head_var, type_var, mask_var)
    # loss_eval = loss_eval.mean()

    # l2 regularization?
    if regular == 'l2':
        l2_penalty = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
        loss_train = loss_train + gamma * l2_penalty

    updates = create_updates(loss_train, network, opt, learning_rate, momentum, beta1, beta2)

    # Compile a function performing a training step on a mini-batch
    train_fn = theano.function([word_var, char_var, pos_var, head_var, type_var, mask_var], loss_train, updates=updates,
                               on_unused_input='warn')
    # Compile a second function evaluating the loss and accuracy of network
    eval_fn = theano.function([word_var, char_var, pos_var, head_var, type_var, mask_var], [loss_eval, energies_eval],
                              on_unused_input='warn')

    # Finally, launch the training loop.
    logger.info("Start training: (#training data: %d, batch size: %d, clip: %.1f)..." % (
        num_data, batch_size, grad_clipping))

    num_batches = num_data / batch_size + 1
    dev_ucorrect = 0.0
    dev_lcorrect = 0.0
    dev_ucorrect_nopunct = 0.0
    dev_lcorrect_nopunct = 0.0
    best_epoch = 0
    test_ucorrect = 0.0
    test_lcorrect = 0.0
    test_ucorrect_nopunct = 0.0
    test_lcorrect_nopunct = 0.0
    test_total = 0
    test_total_nopunc = 0
    test_inst = 0
    lr = learning_rate
    for epoch in range(1, num_epochs + 1):
        print 'Epoch %d (learning rate=%.5f, decay rate=%.4f, beta1=%.3f, beta2=%.3f): ' % (
            epoch, lr, decay_rate, beta1, beta2)
        train_err = 0.0
        train_inst = 0
        start_time = time.time()
        num_back = 0
        for batch in xrange(1, num_batches + 1):
            wids, cids, pids, hids, tids, masks = data_utils.get_batch(data_train, batch_size)
            err = train_fn(wids, cids, pids, hids, tids, masks)
            train_err += err * wids.shape[0]
            train_inst += wids.shape[0]
            time_ave = (time.time() - start_time) / batch
            time_left = (num_batches - batch) * time_ave

            # update log
            sys.stdout.write("\b" * num_back)
            log_info = 'train: %d/%d loss: %.4f, time left: %.2fs' % (
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
            wids, cids, pids, hids, tids, masks = batch
            err, energies = eval_fn(wids, cids, pids, hids, tids, masks)
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

        if dev_ucorrect_nopunct <= dev_ucorr_nopunc:
            dev_ucorrect_nopunct = dev_ucorr_nopunc
            dev_lcorrect_nopunct = dev_lcorr_nopunc
            dev_ucorrect = dev_ucorr
            dev_lcorrect = dev_lcorr
            best_epoch = epoch

            test_err = 0.0
            test_ucorr = 0.0
            test_lcorr = 0.0
            test_ucorr_nopunc = 0.0
            test_lcorr_nopunc = 0.0
            test_total = 0
            test_total_nopunc = 0
            test_inst = 0
            for batch in data_utils.iterate_batch(data_test, batch_size):
                wids, cids, pids, hids, tids, masks = batch
                err, energies = eval_fn(wids, cids, pids, hids, tids, masks)
                test_err += err * wids.shape[0]
                pars_pred, types_pred = parser.decode_MST(energies, masks)
                ucorr, lcorr, total, ucorr_nopunc, \
                lcorr_nopunc, total_nopunc = parser.eval(wids, pids, pars_pred, types_pred, hids, tids, masks,
                                                         tmp_dir + '/test_parse%d' % epoch, word_alphabet, pos_alphabet,
                                                         type_alphabet, punct_set=punct_set)
                test_inst += wids.shape[0]

                test_ucorr += ucorr
                test_lcorr += lcorr
                test_total += total

                test_ucorr_nopunc += ucorr_nopunc
                test_lcorr_nopunc += lcorr_nopunc
                test_total_nopunc += total_nopunc
            test_ucorrect = test_ucorr
            test_lcorrect = test_lcorr
            test_ucorrect_nopunct = test_ucorr_nopunc
            test_lcorrect_nopunct = test_lcorr_nopunc

        print 'best dev  W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%% (epoch: %d)' % (
            dev_ucorrect, dev_lcorrect, dev_total, dev_ucorrect * 100 / dev_total, dev_lcorrect * 100 / dev_total,
            best_epoch)
        print 'best dev  Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%% (epoch: %d)' % (
            dev_ucorrect_nopunct, dev_lcorrect_nopunct, dev_total_nopunc, dev_ucorrect_nopunct * 100 / dev_total_nopunc,
            dev_lcorrect_nopunct * 100 / dev_total_nopunc, best_epoch)
        print 'best test W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%% (epoch: %d)' % (
            test_ucorrect, test_lcorrect, test_total, test_ucorrect * 100 / test_total,
            test_lcorrect * 100 / test_total, best_epoch)
        print 'best test Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%% (epoch: %d)' % (
            test_ucorrect_nopunct, test_lcorrect_nopunct, test_total_nopunc,
            test_ucorrect_nopunct * 100 / test_total_nopunc, test_lcorrect_nopunct * 100 / test_total_nopunc,
            best_epoch)

        if epoch in schedule:
        # if epoch % schedule == 0:
            lr = lr * decay_rate
            # lr = learning_rate / (1.0 + epoch * decay_rate)
            updates = create_updates(loss_train, network, opt, lr, momentum, beta1, beta2)
            train_fn = theano.function([word_var, char_var, pos_var, head_var, type_var, mask_var], loss_train,
                                       updates=updates, on_unused_input='warn')


if __name__ == '__main__':
    main()
