__author__ = 'max'

import theano.tensor as T

from lasagne import init
from lasagne.theano_extensions import conv
from lasagne.layers import Conv1DLayer, Layer, InputLayer, helper
from lasagne import nonlinearities

__all__ = [
    "ConvTimeStep1DLayer",
]


class ConvTimeStep1DLayer(Layer):
    """
    CNN with time step at axis=1. The input shape should be [batch_size, n-step, num_input_channels, input_length].

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 3D tensor, with shape
        ``(batch_size, n-step, num_input_channels, input_length)``.

    num_filters : int
        The number of learnable convolutional filters this layer has.

    filter_size : int or iterable of int
        An integer or a 1-element tuple specifying the size of the filters.

    stride : int or iterable of int
        An integer or a 1-element tuple specifying the stride of the
        convolution operation.

    pad : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        An integer or a 1-element tuple results in symmetric zero-padding of
        the given size on both borders.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size (rounded down) on both sides.
        When ``stride=1`` this results in an output size equal to the input
        size. Even filter size is not supported.

        ``'valid'`` is an alias for ``0`` (no padding / a valid convolution).

    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).

        If True, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be a
        matrix (2D).

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a 3D tensor with shape
        ``(num_filters, num_input_channels, filter_length)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, input_length)`` instead.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    flip_filters : bool (default: True)
        Whether to flip the filters before sliding them over the input,
        performing a convolution (this is the default), or not to flip them and
        perform a correlation. Note that for some other convolutional layers in
        Lasagne, flipping incurs an overhead and is disabled by default --
        check the documentation when using learned weights from another layer.

    convolution : callable
        The convolution implementation to use. The
        `lasagne.theano_extensions.conv` module provides some alternative
        implementations for 1D convolutions, because the Theano API only
        features a 2D convolution implementation. Usually it should be fine
        to leave this at the default value. Note that not all implementations
        support all settings for `pad` and `subsample`.

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.

    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.

    b : Theano shared variable or expression
        Variable or expression representing the biases.
    """

    def __init__(self, incoming, num_filters, filter_size, stride=1,
                 pad=0, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, flip_filters=True,
                 convolution=conv.conv1d_mc0, **kwargs):
        if isinstance(incoming, tuple):
            input_shape = incoming
        else:
            input_shape = incoming.output_shape

        # Retrieve the supplied name, if it exists; otherwise use ''
        if 'name' in kwargs:
            basename = kwargs['name'] + '.'
            # Create a separate version of kwargs for the contained layers
            # which does not include 'name'
            layer_kwargs = dict((key, arg) for key, arg in kwargs.items() if key != 'name')
        else:
            basename = ''
            layer_kwargs = kwargs
        self.conv1d = Conv1DLayer(InputLayer((None,) + input_shape[2:]), num_filters, filter_size, stride, pad,
                                  untie_biases, W, b, nonlinearity, flip_filters, convolution, name=basename + "conv1d",
                                  **layer_kwargs)
        self.W = self.conv1d.W
        self.b = self.conv1d.b
        super(ConvTimeStep1DLayer, self).__init__(incoming, **kwargs)

    def get_params(self, **tags):
        # Get all parameters from this layer, the master layer
        params = super(ConvTimeStep1DLayer, self).get_params(**tags)
        # Combine with all parameters from the child layers
        params += helper.get_all_params(self.conv1d, **tags)
        return params

    def get_output_shape_for(self, input_shape):
        batch_size = input_shape[0]
        time_steps = input_shape[1]

        new_batch_size = batch_size * time_steps if batch_size and time_steps else batch_size
        input_shape = (new_batch_size, ) + input_shape[2:]
        output_shape = self.conv1d.get_output_shape_for(input_shape)
        return (batch_size, time_steps) + output_shape[1:]

    def get_output_for(self, input, **kwargs):
        # [batch, n-step, num_input_channels, input_length]
        input_shape = input.shape

        batch_size = input_shape[0]
        time_steps = input_shape[1]

        # [batch * n-step, num_input_channels, input_length]
        input_shape = (batch_size * time_steps, input_shape[2], input_shape[3])
        output = self.conv1d.get_output_for(T.reshape(input, input_shape), **kwargs)

        # [batch * n-step, num_filters, output_length]
        output_shape = output.shape
        # [batch, n-step, num_filters, output_length]
        output_shape = (batch_size, time_steps, output_shape[1], output_shape[2])
        return T.reshape(output, output_shape)
