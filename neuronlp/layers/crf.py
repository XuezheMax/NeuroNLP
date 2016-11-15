__author__ = 'max'

import theano.tensor as T

from lasagne import init
from lasagne.layers import MergeLayer

__all__ = [
    "ChainCRFLayer",
    "TreeAffineCRFLayer",
    "TreeBiAffineCRFLayer",
]


class ChainCRFLayer(MergeLayer):
    """
    ChainCRFLayer(incoming, num_labels, mask_input=None, W=init.GlorotUniform(), b=init.Constant(0.), **kwargs)

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
        The output of this layer should be a 3D tensor with shape
        ``(batch_size, input_length, num_input_features)``
    num_labels : int
        The number of labels of the crf layer
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a tensor with shape ``(num_inputs, num_units)``,
        where ``num_inputs`` is the size of the second dimension of the input.
        See :func:`lasagne.utils.create_param` for more information.
    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_units,)
    """

    def __init__(self, incoming, num_labels, mask_input=None, W=init.GlorotUniform(), b=init.Constant(0.), **kwargs):
        # This layer inherits from a MergeLayer, because it can have two
        # inputs - the layer input, and the mask.
        # We will just provide the layer input as incomings, unless a mask input was provided.

        self.input_shape = incoming.output_shape
        incomings = [incoming]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = 1

        super(ChainCRFLayer, self).__init__(incomings, **kwargs)
        self.num_labels = num_labels + 1
        self.pad_label_index = num_labels

        num_inputs = self.input_shape[2]
        self.W = self.add_param(W, (num_inputs, self.num_labels, self.num_labels), name="W")

        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (self.num_labels, self.num_labels), name="b", regularizable=False)

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        return input_shape[0], input_shape[1], self.num_labels, self.num_labels

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable.

        Parameters
        ----------
        :param inputs: list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``.
        :return: theano.TensorType
            Symbolic output variable.
        """
        input = inputs[0]
        mask = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]

        # compute out by tensor dot ([batch, length, input] * [input, num_label, num_label]
        # the shape of out should be [batch, length, num_label, num_label]
        out = T.tensordot(input, self.W, axes=[[2], [0]])

        if self.b is not None:
            b_shuffled = self.b.dimshuffle('x', 'x', 0, 1)
            out = out + b_shuffled

        if mask is not None:
            mask_shuffled = mask.dimshuffle(0, 1, 'x', 'x')
            out = out * mask_shuffled
        return out


class TreeAffineCRFLayer(MergeLayer):
    """
    TreeAffineCRFLayer(incoming, num_labels, mask_input=None, W_h=init.GlorotUniform(), W_c=init.GlorotUniform(),
        b=init.Constant(0.), **kwargs)
    """

    def __init__(self, incoming, num_labels, mask_input=None, W_h=init.GlorotUniform(), W_c=init.GlorotUniform(),
                 b=init.Constant(0.), **kwargs):
        # This layer inherits from a MergeLayer, because it can have two
        # inputs - the layer input, and the mask.
        # We will just provide the layer input as incomings, unless a mask input was provided.
        self.input_shape = incoming.output_shape
        incomings = [incoming]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = 1

        super(TreeAffineCRFLayer, self).__init__(incomings, **kwargs)
        self.num_labels = num_labels
        dim_inputs = self.input_shape[2]

        # add parameters
        self.W_h = self.add_param(W_h, (dim_inputs, self.num_labels), name='W_h')

        self.W_c = self.add_param(W_c, (dim_inputs, self.num_labels), name='W_c')

        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (self.num_labels,), name='b', regularizable=False)

    def get_output_shape_for(self, input_shapes):
        """

        :param input_shapes:
        :return: the shape of output [batch_size, length, length, num_labels]
        """
        input_shape = input_shapes[0]
        return input_shape[0], input_shape[1], input_shape[1], self.num_labels

    def get_output_for(self, inputs, **kwargs):
        """

        :param inputs: inputs: list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``.
        :return: theano.TensorType
            Symbolic output variable.
        """
        input = inputs[0]
        mask = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]

        # compute head part by tensor dot ([batch, length, dim] * [dim, num_label]
        # the shape of s_h should be [batch, length, num_label]
        s_h = T.tensordot(input, self.W_h, axes=[[2], [0]])

        if self.b is not None:
            b_shuffled = self.b.dimshuffle('x', 'x', 0)
            s_h = s_h + b_shuffled

        # compute child part by tensor dot ([batch, length, dim] * [dim, num_label]
        # the shape of s_c should be [batch, length, num_label]
        s_c = T.tensordot(input, self.W_c, axes=[[2], [0]])

        # compute out
        input_shape = input.shape
        # output shape = [batch, length, length, num_label]
        out = T.cast(T.alloc(0.0, input_shape[0], input_shape[1], input_shape[1], self.num_labels), 'floatX')
        out = out + s_h.dimshuffle(0, 1, 'x', 2)
        out = out + s_c.dimshuffle(0, 'x', 1, 2)

        if mask is not None:
            mask_shuffled = mask.dimshuffle(0, 1, 'x', 'x')
            out = out * mask_shuffled
            mask_shuffled = mask.dimshuffle(0, 'x', 1, 'x')
            out = out * mask_shuffled
        return out


class TreeBiAffineCRFLayer(MergeLayer):
    """
    TreeBiAffineCRFLayer(incoming, num_labels, mask_input=None, U=init.GlorotUniform(), W_h=init.GlorotUniform(),
        b=init.Constant(0.), **kwargs)
    """

    def __init__(self, incoming, num_labels, mask_input=None, U=init.GlorotUniform(), W_h=init.GlorotUniform(),
                 W_c=init.GlorotUniform(), b=init.Constant(0.), **kwargs):
        # This layer inherits from a MergeLayer, because it can have two
        # inputs - the layer input, and the mask.
        # We will just provide the layer input as incomings, unless a mask input was provided.
        self.input_shape = incoming.output_shape
        incomings = [incoming]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = 1

        super(TreeBiAffineCRFLayer, self).__init__(incomings, **kwargs)
        self.num_labels = num_labels
        dim_inputs = self.input_shape[2]

        # add parameters
        self.U = self.add_param(U, (dim_inputs, dim_inputs, self.num_labels), name='U')
        self.W_h = None if W_h is None else self.add_param(W_h, (dim_inputs, self.num_labels), name='W_h')
        self.W_c = None if W_c is None else self.add_param(W_c, (dim_inputs, self.num_labels), name='W_c')

        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (self.num_labels,), name='b', regularizable=False)

    def get_output_shape_for(self, input_shapes):
        """

        :param input_shapes:
        :return: the shape of output [batch_size, length, length, num_labels]
        """
        input_shape = input_shapes[0]
        return input_shape[0], input_shape[1], input_shape[1], self.num_labels

    def get_output_for(self, inputs, **kwargs):
        """

        :param inputs: inputs: list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``.
        :return: theano.TensorType
            Symbolic output variable.
        """
        input = inputs[0]
        mask = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]

        # compute the bi-affine part
        # first via tensor dot ([batch, length, dim] * [dim, dim, num_label])
        # output shape = [batch, length, dim, num_label]
        out = T.tensordot(input, self.U, axes=[[2], [0]])
        # second via tensor dot ([batch, length, dim, num_label] * [batch, dim, length)
        # output shape = [batch, length, length, num_label]
        out = T.batched_tensordot(out, input.dimshuffle(0, 2, 1), axes=([2], [1]))
        out = out.dimshuffle(0, 1, 3, 2)

        # compute head bias part by tensor dot ([batch, length, dim] * [dim, num_label])
        # the shape of s_h should be [batch, length, num_label]
        if self.W_h is not None:
            s_h = T.tensordot(input, self.W_h, axes=[[2], [0]])
            out = out + s_h.dimshuffle(0, 1, 'x', 2)

        # compute child part by tensor dot ([batch, length, dim] * [dim, num_label]
        # the shape of s_c should be [batch, length, num_label]
        if self.W_c is not None:
            s_c = T.tensordot(input, self.W_c, axes=[[2], [0]])
            out = out + s_c.dimshuffle(0, 'x', 1, 2)

        # add bias part.
        if self.b is not None:
            out = out + self.b.dimshuffle('x', 'x', 'x', 0)

        if mask is not None:
            mask_shuffled = mask.dimshuffle(0, 1, 'x', 'x')
            out = out * mask_shuffled
            mask_shuffled = mask.dimshuffle(0, 'x', 1, 'x')
            out = out * mask_shuffled
        return out
