__author__ = 'max'

import theano.tensor as T

from lasagne.utils import as_tuple
from lasagne.layers import Pool1DLayer

__all__ = [
    "PoolTimeStep1DLayer",
]


class PoolTimeStep1DLayer(Pool1DLayer):
    """
    Pool with time step at axis=1. The input shape should be [batch_size, n-step, num_input_channels, input_length].
    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    pool_size : integer or iterable
        The length of the pooling region. If an iterable, it should have a
        single element.

    stride : integer, iterable or ``None``
        The stride between sucessive pooling regions.
        If ``None`` then ``stride == pool_size``.

    pad : integer or iterable
        The number of elements to be added to the input on each side.
        Must be less than stride.

    ignore_border : bool
        If ``True``, partial pooling regions will be ignored.
        Must be ``True`` if ``pad != 0``.

    mode : {'max', 'average_inc_pad', 'average_exc_pad'}
        Pooling mode: max-pooling or mean-pooling including/excluding zeros
        from partially padded pooling regions. Default is 'max'.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    """

    def __init__(self, incoming, pool_size, stride=None, pad=0,
                 ignore_border=True, mode='max', **kwargs):
        super(Pool1DLayer, self).__init__(incoming, **kwargs)

        if len(self.input_shape) != 4:
            raise ValueError("Tried to create a 1D time-step pooling layer with "
                             "input shape %r. Expected 4 input dimensions "
                             "(batchsize, n-step, channels, 1 spatial dimensions)."
                             % (self.input_shape,))

        self.pool_size = as_tuple(pool_size, 1)
        self.stride = self.pool_size if stride is None else as_tuple(stride, 1)
        self.pad = as_tuple(pad, 1)
        self.ignore_border = ignore_border
        self.mode = mode

    def get_output_for(self, input, **kwargs):
        # [batch, n-step, num_input_channels, input_length]
        input_shape = input.shape

        batch_size = input_shape[0]
        time_steps = input_shape[1]

        # [batch * n-step, num_input_channels, input_length]
        input_shape = (batch_size * time_steps, input_shape[2], input_shape[3])
        output = super(PoolTimeStep1DLayer, self).get_output_for(T.reshape(input, input_shape), **kwargs)

        # [batch * n-step, num_input_channels, pool_length]
        output_shape = output.shape
        # [batch, n-step, num_input_channels, pool_length]
        output_shape = (batch_size, time_steps, output_shape[1], output_shape[2])
        return T.reshape(output, output_shape)
