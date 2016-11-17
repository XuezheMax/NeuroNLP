__author__ = 'max'

import numpy

import theano
import theano.tensor as T
from theano.tensor import as_tensor_variable
from theano.gof import Op, Apply
from theano.tensor.nlinalg import matrix_inverse

__all__ = [
    "LogAbsDet",
    "logabsdet",
    "LogSafeAbsDet",
    "logabsdet_safe",
    "theano_logsumexp",
]


class LogAbsDet(Op):
    """
    Computes the logarithm of absolute determinants of a sequence of square
    matrix M, log(abs(det(M))), on CPU. Avoids det(M) overflow/
    underflow.

    TODO: add GPU code!
    """

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        o = theano.tensor.scalar(dtype=x.dtype)
        return Apply(self, [x], [o])

    def perform(self, node, inputs, outputs, params=None):
        try:
            (x,) = inputs
            (z,) = outputs
            s = numpy.linalg.svd(x, compute_uv=False)
            log_abs_det = numpy.sum(numpy.log(numpy.abs(s)))
            z[0] = numpy.asarray(log_abs_det, dtype=x.dtype)
        except Exception:
            print('Failed to compute logabsdet of {}.'.format(x))
            raise

    def grad(self, inputs, g_outputs):
        [gz] = g_outputs
        [x] = inputs
        return [gz * matrix_inverse(x).T]

    def __str__(self):
        return "LogAbsDet"


logabsdet = LogAbsDet()


class LogSafeAbsDet(Op):
    """
    Computes the logarithm of absolute determinants of a sequence of square
    matrix M, log(abs(det(M))), on CPU. Avoids det(M) overflow/
    underflow.

    Make the matrix inverse numerical safe in the computation of gradient.

    TODO: add GPU code!
    """

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        o = theano.tensor.scalar(dtype=x.dtype)
        return Apply(self, [x], [o])

    def perform(self, node, inputs, outputs, params=None):
        try:
            (x,) = inputs
            (z,) = outputs
            s = numpy.linalg.svd(x, compute_uv=False)
            log_abs_det = numpy.sum(numpy.log(numpy.abs(s)))
            z[0] = numpy.asarray(log_abs_det, dtype=x.dtype)
        except Exception:
            print('Failed to compute logabsdet of {}.'.format(x))
            raise

    def grad(self, inputs, g_outputs):
        [gz] = g_outputs
        [x] = inputs
        UPPER_BOUND = 1e+38
        MAX_ELEM = 1e+10
        epsilon = 1e-7
        dtype = numpy.dtype(theano.config.floatX).type

        x_inv = matrix_inverse(x)
        x_inv = T.switch(T.isnan(x_inv), 0, x_inv)
        x_inv = T.clip(x_inv, dtype(-UPPER_BOUND), dtype(UPPER_BOUND))
        max_element = (T.abs_(x_inv)).max()
        target = T.clip(max_element, 0, dtype(MAX_ELEM))
        multiplier = target / (dtype(epsilon) + max_element)
        x_inv = x_inv * multiplier
        return [gz * x_inv.T]

    def __str__(self):
        return "LogSafeAbsDet"


logabsdet_safe = LogSafeAbsDet()


def theano_logsumexp(x, axis=None):
    """
    Compute log(sum(exp(x), axis=axis) in a numerically stable
    fashion.
    Parameters
    ----------
    x : tensor_like
        A Theano tensor (any dimension will do).
    axis : int or symbolic integer scalar, or None
        Axis over which to perform the summation. `None`, the
        default, performs over all axes.
    Returns
    -------
    result : ndarray or scalar
        The result of the log(sum(exp(...))) operation.
    """

    xmax = x.max(axis=axis, keepdims=True)
    xmax_ = x.max(axis=axis)
    return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))
