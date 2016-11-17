__author__ = 'max'

import numpy

import theano
import theano.tensor as T
from theano.tensor import as_tensor_variable
from theano.gof import Op, Apply
from theano.tensor.nlinalg import matrix_inverse, matrix_dot

__all__ = [
    'MatrixScaledInverse',
    'matrix_scaled_inverse',
    "LogAbsDet",
    "logabsdet",
    "LogSafeAbsDet",
    "logabsdet_safe",
    "theano_logsumexp",
]


class MatrixScaledInverse(Op):
    """Computes the inverse of a matrix :math:`A`.

    Given a square matrix :math:`A`, ``matrix_inverse`` returns a square
    matrix :math:`A_{inv}` such that the dot product :math:`A \cdot A_{inv}`
    and :math:`A_{inv} \cdot A` equals the identity matrix :math:`I`.

    Notes
    -----
    When possible, the call to this op will be optimized to the call
    of ``solve``.

    """

    __props__ = ()

    def __init__(self):
        pass

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs, params=None):
        (x,) = inputs
        (z,) = outputs
        UPPER_BOUND = 1e+38
        MAX_ELEM = 1e+10
        epsilon = 1e-7
        dtype = numpy.dtype(theano.config.floatX).type
        x_inv = numpy.linalg.inv(x).astype(x.dtype)
        # x_inv = numpy.where(numpy.isnan(x_inv), 0, x_inv)
        x_inv = numpy.clip(x_inv, dtype(-UPPER_BOUND), dtype(UPPER_BOUND))
        max_element = (numpy.abs(x_inv)).max()
        target = numpy.clip(max_element, 0, dtype(MAX_ELEM))
        multiplier = target / (dtype(epsilon) + max_element)
        z[0] = x_inv * multiplier

    def grad(self, inputs, g_outputs):
        r"""The gradient function should return

            .. math:: V\frac{\partial X^{-1}}{\partial X},

        where :math:`V` corresponds to ``g_outputs`` and :math:`X` to
        ``inputs``. Using the `matrix cookbook
        <http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3274>`_,
        one can deduce that the relation corresponds to

            .. math:: (X^{-1} \cdot V^{T} \cdot X^{-1})^T.

        """
        x, = inputs
        xi = matrix_inverse(x)
        gz, = g_outputs
        # TT.dot(gz.T,xi)
        return [-matrix_dot(xi, gz.T, xi).T]

    def R_op(self, inputs, eval_points):
        r"""The gradient function should return

            .. math:: \frac{\partial X^{-1}}{\partial X}V,

        where :math:`V` corresponds to ``g_outputs`` and :math:`X` to
        ``inputs``. Using the `matrix cookbook
        <http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3274>`_,
        one can deduce that the relation corresponds to

            .. math:: X^{-1} \cdot V \cdot X^{-1}.

        """
        x, = inputs
        xi = matrix_inverse(x)
        ev, = eval_points
        if ev is None:
            return [None]
        return [-matrix_dot(xi, ev, xi)]

    def infer_shape(self, node, shapes):
        return shapes


matrix_scaled_inverse = MatrixScaledInverse()


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
        return [gz * matrix_scaled_inverse(x).T]

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
