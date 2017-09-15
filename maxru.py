import time
import sys
import argparse

import numpy as np
import theano.tensor as T
import theano
import lasagne

import lasagne.nonlinearities as nonlinearities
from neuronlp.layers import MAXRULayer
from lasagne.layers import Gate, DenseLayer

BATCH_SIZE = 128


def get_batch(batch_size, pos, binominal, length):
    if binominal:
        x = np.random.binomial(1, 0.5, [batch_size, length])
        x = 2 * x - 1
    else:
        x = np.random.uniform(-1.0, 1.0, [batch_size, length])
    y = np.zeros((batch_size,), dtype=np.int32)
    y[:] = (x[:, pos]) > 0.0
    return np.reshape(x, [batch_size, length, 1]).astype(np.float32), y


def train(layer_output, input_var, target_var, batch_size, length, position, binominal):
    predictions = lasagne.layers.get_output(layer_output)
    acc = lasagne.objectives.binary_accuracy(predictions, target_var)
    acc = acc.sum()

    loss = lasagne.objectives.binary_crossentropy(predictions, target_var)
    loss = loss.sum()

    learning_rate = 0.001
    steps_per_epoch = 1000
    params = lasagne.layers.get_all_params(layer_output, trainable=True)
    # updates = lasagne.updates.sgd(loss, params=params, learning_rate=learning_rate)
    updates = lasagne.updates.adam(loss, params=params, learning_rate=learning_rate)
    train_fn = theano.function([input_var, target_var], [loss, acc, predictions], updates=updates)

    num_epoches = 500
    accuracies = np.zeros(num_epoches)
    for epoch in xrange(num_epoches):
        start_time = time.time()
        print 'Epoch %d (learning rate=%.4f)' % (epoch, learning_rate)
        loss = 0.0
        correct = 0.0
        num_back = 0
        for step in xrange(steps_per_epoch):
            x, y = get_batch(batch_size, position, binominal, length)
            err, corr, pred = train_fn(x, y)
            # print x
            # print y
            # print pred
            loss += err
            correct += corr
            num_inst = (step + 1) * batch_size

        assert num_inst == batch_size * steps_per_epoch
        accuracies[epoch] = correct * 100 / num_inst
        print 'inst: %d loss: %.4f, corr: %d, acc: %.2f%%, time: %.2fs' % (
            num_inst, loss / num_inst, correct, correct * 100 / num_inst, time.time() - start_time)

    return accuracies[-50:].mean()


def exe_maxru(length, num_units, position, binominal):
    batch_size = BATCH_SIZE

    input_var = T.tensor3(name='inputs', dtype=theano.config.floatX)
    target_var = T.ivector(name='targets')

    layer_input = lasagne.layers.InputLayer(shape=(None, length, 1), input_var=input_var, name='input')

    time_updategate = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None)

    time_update = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                       b=lasagne.init.Constant(0.), nonlinearity=nonlinearities.tanh)

    resetgate = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                     W_cell=lasagne.init.GlorotUniform())

    updategate = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                      W_cell=lasagne.init.GlorotUniform())

    hiden_update = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                        b=lasagne.init.Constant(0.), nonlinearity=nonlinearities.tanh)

    layer_taru = MAXRULayer(layer_input, num_units, max_length=length,
                            P_time=lasagne.init.GlorotUniform(), nonlinearity=nonlinearities.tanh,
                            resetgate=resetgate, updategate=updategate, hidden_update=hiden_update,
                            time_updategate=time_updategate, time_update=time_update,
                            only_return_final=True, name='MAXRU', p=0.)

    W = layer_taru.W_hid_to_hidden_update.sum()
    U = layer_taru.W_in_to_hidden_update.sum()
    b = layer_taru.b_hidden_update.sum()

    layer_output = DenseLayer(layer_taru, num_units=1, nonlinearity=nonlinearities.sigmoid, name='output')

    return train(layer_output, input_var, target_var, W, U, b, batch_size, length, position, binominal)


def main():
    parser = argparse.ArgumentParser(description='Tuning with bi-directional RNN')
    parser.add_argument('--num_units', type=int, default=2, help='Number of units')
    parser.add_argument('--binominal', action='store_true',
                        help='If the data are sampled from bi-nominal distribution.')
    args = parser.parse_args()

    NUM_UNITS = args.num_units
    BINOMINAL = args.binominal

    filename = 'tmp/%s.%s.dim=%d' % ('binominal' if BINOMINAL else 'uniform', 'maxru', NUM_UNITS)
    fp = open(filename, 'w')
    print 'data: %s' % ('binominal' if BINOMINAL else 'uniform')
    num_runs = 100
    exe = exe_maxru
    for length in [20, 40, 50]:
        result = 0.
        position = 0
        print 'architecture: %s (dim=%d, length=%d, postion=%d)' % ('maxru', NUM_UNITS, length, position)
        fp.write('length=%d, pos=%d:\n' % (length, position))
        fp.flush()
        for run in xrange(num_runs):
            acc = exe(length, NUM_UNITS, position, BINOMINAL)
            fp.write('%.2f, ' % acc)
            result = result + acc
            fp.flush()

        fp.write('%.2f\n\n' % (result / num_runs))
        fp.flush()

        result = 0.
        position = (length - 1) / 2
        print 'architecture: %s (dim=%d, length=%d, postion=%d)' % ('taru', NUM_UNITS, length, position)
        fp.write('length=%d, pos=%d:\n' % (length, position))
        fp.flush()
        for run in xrange(num_runs):
            acc = exe(length, NUM_UNITS, position, BINOMINAL)
            fp.write('%.2f, ' % acc)
            result = result + acc
            fp.flush()

        fp.write('%.2f\n\n' % (result / num_runs))
        fp.flush()

    fp.close()


if __name__ == '__main__':
    main()
