__author__ = 'max'
"""
Layers to construct recurrent networks. Recurrent layers can be used similarly
to feed-forward layers except that the input shape is expected to be
``(batch_size, sequence_length, num_inputs)``.

The implementation
"""
import numpy as np
import theano
import theano.tensor as T
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan
from lasagne.random import get_rng
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from lasagne.layers import MergeLayer, Layer, InputLayer, DenseLayer, Gate
from lasagne.layers import helper

__all__ = [
    "CustomRecurrentLayer",
    "RecurrentLayer",
    "LSTMLayer",
    "GRULayer",
    "SGRULayer",
    "MAXRULayer"
]


class CustomRecurrentLayer(MergeLayer):
    """
    Coustom Recurrent Layer based on lasagne.layer.class CustomRecurrentLayer, with recurrent dropout options

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    input_to_hidden : :class:`lasagne.layers.Layer`
        :class:`lasagne.layers.Layer` instance which connects input to the
        hidden state (:math:`f_i`).  This layer may be connected to a chain of
        layers, which must end in a :class:`lasagne.layers.InputLayer` with the
        same input shape as `incoming`, except for the first dimension: When
        ``precompute_input == True`` (the default), it must be
        ``incoming.output_shape[0]*incoming.output_shape[1]`` or ``None``; when
        ``precompute_input == False``, it must be ``incoming.output_shape[0]``
        or ``None``.
    hidden_to_hidden : :class:`lasagne.layers.Layer`
        Layer which connects the previous hidden state to the new state
        (:math:`f_h`).  This layer may be connected to a chain of layers, which
        must end in a :class:`lasagne.layers.InputLayer` with the same input
        shape as `hidden_to_hidden`'s output shape.
    nonlinearity : callable or None
        Nonlinearity to apply when computing new state (:math:`\sigma`). If
        None is provided, no nonlinearity will be applied.
    hid_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial hidden state (:math:`h_0`).
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.
    p : float or scalar tensor
        The probability of recurrent dropout (if p=0, no dropout is applied)
    """

    def __init__(self, incoming, input_to_hidden, hidden_to_hidden,
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 p=0.,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have three
        # inputs - the layer input, the mask and the initial hidden state.  We
        # will just provide the layer input as incomings, unless a mask input
        # or initial hidden state was provided.
        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings) - 1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings) - 1

        super(CustomRecurrentLayer, self).__init__(incomings, **kwargs)

        self.input_to_hidden = input_to_hidden
        self.hidden_to_hidden = hidden_to_hidden
        self.learn_init = learn_init
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # Check that the input_to_hidden connection can appropriately handle
        # a first dimension of input_shape[0]*input_shape[1] when we will
        # precompute the input dot product
        if (self.precompute_input and
                    input_to_hidden.output_shape[0] is not None and
                    input_shape[0] is not None and
                    input_shape[1] is not None and
                (input_to_hidden.output_shape[0] !=
                         input_shape[0] * input_shape[1])):
            raise ValueError(
                'When precompute_input == True, '
                'input_to_hidden.output_shape[0] must equal '
                'incoming.output_shape[0]*incoming.output_shape[1] '
                '(i.e. batch_size*sequence_length) or be None but '
                'input_to_hidden.output_shape[0] = {} and '
                'incoming.output_shape[0]*incoming.output_shape[1] = '
                '{}'.format(input_to_hidden.output_shape[0],
                            input_shape[0] * input_shape[1]))

        # Check that the first dimension of input_to_hidden and
        # hidden_to_hidden's outputs match when we won't precompute the input
        # dot product
        if (not self.precompute_input and
                    input_to_hidden.output_shape[0] is not None and
                    hidden_to_hidden.output_shape[0] is not None and
                (input_to_hidden.output_shape[0] !=
                     hidden_to_hidden.output_shape[0])):
            raise ValueError(
                'When precompute_input == False, '
                'input_to_hidden.output_shape[0] must equal '
                'hidden_to_hidden.output_shape[0] but '
                'input_to_hidden.output_shape[0] = {} and '
                'hidden_to_hidden.output_shape[0] = {}'.format(
                    input_to_hidden.output_shape[0],
                    hidden_to_hidden.output_shape[0]))

        # Check that input_to_hidden and hidden_to_hidden output shapes match,
        # but don't check a dimension if it's None for either shape
        if not all(s1 is None or s2 is None or s1 == s2
                   for s1, s2 in zip(input_to_hidden.output_shape[1:],
                                     hidden_to_hidden.output_shape[1:])):
            raise ValueError("The output shape for input_to_hidden and "
                             "hidden_to_hidden must be equal after the first "
                             "dimension, but input_to_hidden.output_shape={} "
                             "and hidden_to_hidden.output_shape={}".format(
                input_to_hidden.output_shape,
                hidden_to_hidden.output_shape))

        # Check that input_to_hidden's output shape is the same as
        # hidden_to_hidden's input shape but don't check a dimension if it's
        # None for either shape
        if not all(s1 is None or s2 is None or s1 == s2
                   for s1, s2 in zip(input_to_hidden.output_shape[1:],
                                     hidden_to_hidden.input_shape[1:])):
            raise ValueError("The output shape for input_to_hidden "
                             "must be equal to the input shape of "
                             "hidden_to_hidden after the first dimension, but "
                             "input_to_hidden.output_shape={} and "
                             "hidden_to_hidden.input_shape={}".format(
                input_to_hidden.output_shape,
                hidden_to_hidden.input_shape))

        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        # Initialize hidden state
        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1,) + hidden_to_hidden.output_shape[1:],
                name="hid_init", trainable=learn_init, regularizable=False)

    def get_params(self, **tags):
        # Get all parameters from this layer, the master layer
        params = super(CustomRecurrentLayer, self).get_params(**tags)
        # Combine with all parameters from the child layers
        params += helper.get_all_params(self.input_to_hidden, **tags)
        params += helper.get_all_params(self.hidden_to_hidden, **tags)
        return params

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return (input_shape[0],) + self.hidden_to_hidden.output_shape[1:]
        # Otherwise, the shape will be (n_batch, n_steps, trailing_dims...)
        else:
            return ((input_shape[0], input_shape[1]) +
                    self.hidden_to_hidden.output_shape[1:])

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable.

        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]

        # Input should be provided as (n_batch, n_time_steps, n_features)
        # but scan requires the iterable dimension to be first
        # So, we need to dimshuffle to (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, *range(2, input.ndim))
        seq_len, num_batch = input.shape[0], input.shape[1]
        dropout_mask = None
        if not deterministic and self.p:
            one = T.constant(1)
            retain_prob = one - self.p
            dropout_mask_shape = (num_batch, self.hidden_to_hidden.output_shape[1])
            dropout_mask = self._srng.binomial(dropout_mask_shape, p=retain_prob, dtype=input.dtype)

        if self.precompute_input:
            # Because the input is given for all time steps, we can precompute
            # the inputs to hidden before scanning. First we need to reshape
            # from (seq_len, batch_size, trailing dimensions...) to
            # (seq_len*batch_size, trailing dimensions...)
            # This strange use of a generator in a tuple was because
            # input.shape[2:] was raising a Theano error
            trailing_dims = tuple(input.shape[n] for n in range(2, input.ndim))
            input = T.reshape(input, (seq_len * num_batch,) + trailing_dims)
            input = helper.get_output(
                self.input_to_hidden, input, **kwargs)

            # Reshape back to (seq_len, batch_size, trailing dimensions...)
            trailing_dims = tuple(input.shape[n] for n in range(1, input.ndim))
            input = T.reshape(input, (seq_len, num_batch) + trailing_dims)

        # We will always pass the hidden-to-hidden layer params to step
        non_seqs = helper.get_all_params(self.hidden_to_hidden)
        # When we are not precomputing the input, we also need to pass the
        # input-to-hidden parameters to step
        if not self.precompute_input:
            non_seqs += helper.get_all_params(self.input_to_hidden)

        # Create single recurrent computation step function
        def step(input_n, hid_previous, *args):
            # Compute the hidden-to-hidden activation
            hid_pre = helper.get_output(self.hidden_to_hidden,
                                        (hid_previous if (deterministic or self.p == 0) else (hid_previous / retain_prob) * dropout_mask),
                                        **kwargs)

            # If the dot product is precomputed then add it, otherwise
            # calculate the input_to_hidden values and add them
            if self.precompute_input:
                hid_pre += input_n
            else:
                hid_pre += helper.get_output(
                    self.input_to_hidden, input_n, **kwargs)

            # Clip gradients
            if self.grad_clipping:
                hid_pre = theano.gradient.grad_clip(
                    hid_pre, -self.grad_clipping, self.grad_clipping)

            hid = self.nonlinearity(hid_pre)
            return hid

        def step_masked(input_n, mask_n, hid_previous, *args):
            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            hid = step(input_n, hid_previous, *args)
            hid_out = T.switch(mask_n, hid, hid_previous)
            return [hid_out]

        if mask is not None:
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        if not isinstance(self.hid_init, Layer):
            # The code below simply repeats self.hid_init num_batch times in
            # its first dimension.  Turns out using a dot product and a
            # dimshuffle is faster than T.repeat.
            dot_dims = (list(range(1, self.hid_init.ndim - 1)) +
                        [0, self.hid_init.ndim - 1])
            hid_init = T.dot(T.ones((num_batch, 1)),
                             self.hid_init.dimshuffle(dot_dims))

        if not deterministic and self.p:
            one = T.constant(1)
            retain_prob = one - self.p
            non_seqs += [dropout_mask, retain_prob]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, *range(2, hid_out.ndim))

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out


class RecurrentLayer(CustomRecurrentLayer):
    """
    RecurrentLayer based on lasagne.layer.RecurrentLayer, with recurrent dropout options.

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden units in the layer.
    W_in_to_hid : Theano shared variable, numpy array or callable
        Initializer for input-to-hidden weight matrix (:math:`W_x`).
    W_hid_to_hid : Theano shared variable, numpy array or callable
        Initializer for hidden-to-hidden weight matrix (:math:`W_h`).
    b : Theano shared variable, numpy array, callable or None
        Initializer for bias vector (:math:`b`). If None is provided there will
        be no bias.
    nonlinearity : callable or None
        Nonlinearity to apply when computing new state (:math:`\sigma`). If
        None is provided, no nonlinearity will be applied.
    hid_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial hidden state (:math:`h_0`).
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.
    p : float or scalar tensor
        The probability of recurrent dropout (if p=0, no dropout is applied)
    """

    def __init__(self, incoming, num_units,
                 W_in_to_hid=init.Uniform(),
                 W_hid_to_hid=init.Uniform(),
                 b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 p=0.,
                 **kwargs):

        if isinstance(incoming, tuple):
            input_shape = incoming
        else:
            input_shape = incoming.output_shape
        # Retrieve the supplied name, if it exists; otherwise use ''
        if 'name' in kwargs:
            basename = kwargs['name'] + '.'
            # Create a separate version of kwargs for the contained layers
            # which does not include 'name'
            layer_kwargs = dict((key, arg) for key, arg in kwargs.items()
                                if key != 'name')
        else:
            basename = ''
            layer_kwargs = kwargs
        # We will be passing the input at each time step to the dense layer,
        # so we need to remove the second dimension (the time dimension)
        in_to_hid = DenseLayer(InputLayer((None,) + input_shape[2:]),
                               num_units, W=W_in_to_hid, b=b,
                               nonlinearity=None,
                               name=basename + 'input_to_hidden',
                               **layer_kwargs)
        # The hidden-to-hidden layer expects its inputs to have num_units
        # features because it recycles the previous hidden state
        hid_to_hid = DenseLayer(InputLayer((None, num_units)),
                                num_units, W=W_hid_to_hid, b=None,
                                nonlinearity=None,
                                name=basename + 'hidden_to_hidden',
                                **layer_kwargs)

        # Make child layer parameters intuitively accessible
        self.W_in_to_hid = in_to_hid.W
        self.W_hid_to_hid = hid_to_hid.W
        self.b = in_to_hid.b

        # Just use the CustomRecurrentLayer with the DenseLayers we created
        super(RecurrentLayer, self).__init__(
            incoming, in_to_hid, hid_to_hid, nonlinearity=nonlinearity,
            hid_init=hid_init, backwards=backwards, learn_init=learn_init,
            gradient_steps=gradient_steps,
            grad_clipping=grad_clipping, unroll_scan=unroll_scan,
            precompute_input=precompute_input, mask_input=mask_input,
            only_return_final=only_return_final, p=p, **kwargs)


class LSTMLayer(MergeLayer):
    """
    LSTMLayer based on lasagne.layer.LSTM, with recurrent dropout options.

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden/cell units in the layer.
    ingate : Gate
        Parameters for the input gate (:math:`i_t`): :math:`W_{xi}`,
        :math:`W_{hi}`, :math:`w_{ci}`, :math:`b_i`, and :math:`\sigma_i`.
    forgetgate : Gate
        Parameters for the forget gate (:math:`f_t`): :math:`W_{xf}`,
        :math:`W_{hf}`, :math:`w_{cf}`, :math:`b_f`, and :math:`\sigma_f`.
    cell : Gate
        Parameters for the cell computation (:math:`c_t`): :math:`W_{xc}`,
        :math:`W_{hc}`, :math:`b_c`, and :math:`\sigma_c`.
    outgate : Gate
        Parameters for the output gate (:math:`o_t`): :math:`W_{xo}`,
        :math:`W_{ho}`, :math:`w_{co}`, :math:`b_o`, and :math:`\sigma_o`.
    nonlinearity : callable or None
        The nonlinearity that is applied to the output (:math:`\sigma_h`). If
        None is provided, no nonlinearity will be applied.
    cell_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial cell state (:math:`c_0`).
    hid_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial hidden state (:math:`h_0`).
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned.
    peepholes : bool
        If True, the LSTM uses peephole connections.
        When False, `ingate.W_cell`, `forgetgate.W_cell` and
        `outgate.W_cell` are ignored.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.
    p : float or scalar tensor
        The probability of recurrent dropout (if p=0, no dropout is applied)
    """

    def __init__(self, incoming, num_units,
                 ingate=Gate(),
                 forgetgate=Gate(),
                 cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=Gate(),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 p=0.,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have four
        # inputs - the layer input, the mask, the initial hidden state and the
        # inital cell state. We will just provide the layer input as incomings,
        # unless a mask input, inital hidden state or initial cell state was
        # provided.
        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        self.cell_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings) - 1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings) - 1
        if isinstance(cell_init, Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings) - 1

        # Initialize parent layer
        super(LSTMLayer, self).__init__(incomings, **kwargs)

        # If the provided nonlinearity is None, make it linear
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in parameters from the supplied Gate instances
        (self.W_in_to_ingate, self.W_hid_to_ingate, self.b_ingate,
         self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')

        (self.W_in_to_forgetgate, self.W_hid_to_forgetgate, self.b_forgetgate,
         self.nonlinearity_forgetgate) = add_gate_params(forgetgate,
                                                         'forgetgate')

        (self.W_in_to_cell, self.W_hid_to_cell, self.b_cell,
         self.nonlinearity_cell) = add_gate_params(cell, 'cell')

        (self.W_in_to_outgate, self.W_hid_to_outgate, self.b_outgate,
         self.nonlinearity_outgate) = add_gate_params(outgate, 'outgate')

        # If peephole (cell to gate) connections were enabled, initialize
        # peephole connections.  These are elementwise products with the cell
        # state, so they are represented as vectors.
        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(
                ingate.W_cell, (num_units,), name="W_cell_to_ingate")

            self.W_cell_to_forgetgate = self.add_param(
                forgetgate.W_cell, (num_units,), name="W_cell_to_forgetgate")

            self.W_cell_to_outgate = self.add_param(
                outgate.W_cell, (num_units,), name="W_cell_to_outgate")

        # Setup initial values for the cell and the hidden units
        if isinstance(cell_init, Layer):
            self.cell_init = cell_init
        else:
            self.cell_init = self.add_param(
                cell_init, (1, num_units), name="cell_init",
                trainable=learn_init, regularizable=False)

        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable

        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When the cell state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When both the cell state and the hidden state are
            being pre-filled `inputs[-2]` is the hidden state, while
            `inputs[-1]` is the cell state.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        cell_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape
        # create dropout mask
        dropout_mask = None
        if not deterministic and self.p:
            one = T.constant(1)
            retain_prob = one - self.p
            dropout_mask_shape = (num_batch, self.num_units)
            dropout_mask = self._srng.binomial(dropout_mask_shape, p=retain_prob, dtype=input.dtype)

        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell, self.W_in_to_outgate], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)

        # Stack biases into a (4*num_units) vector
        b_stacked = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0)

        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # W_in_stacked is (n_features, 4*num_units). input is then
            # (n_time_steps, n_batch, 4*num_units).
            input = T.dot(input, W_in_stacked) + b_stacked

        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            s = x[:, n * self.num_units:(n + 1) * self.num_units]
            if self.num_units == 1:
                s = T.addbroadcast(s, 1)  # Theano cannot infer this by itself
            return s

        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, cell_previous, hid_previous, *args):
            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # Calculate gates pre-activations and slice
            gates = input_n + T.dot((hid_previous if (deterministic or self.p == 0) else (hid_previous / retain_prob) * dropout_mask),
                                    W_hid_stacked)

            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous * self.W_cell_to_ingate
                forgetgate += cell_previous * self.W_cell_to_forgetgate

            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            # Compute new cell value
            cell = forgetgate * cell_previous + ingate * cell_input

            if self.peepholes:
                outgate += cell * self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            hid = outgate * self.nonlinearity(cell)
            return [cell, hid]

        def step_masked(input_n, mask_n, cell_previous, hid_previous, *args):
            cell, hid = step(input_n, cell_previous, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)

            return [cell, hid]

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            cell_init = T.dot(ones, self.cell_init)

        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(ones, self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]
        # The "peephole" weight matrices are only used when self.peepholes=True
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]

        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        if not deterministic and self.p:
            one = T.constant(1)
            retain_prob = one - self.p
            non_seqs += [dropout_mask, retain_prob]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            cell_out, hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            cell_out, hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out


class GRULayer(MergeLayer):
    """
    GRULayer based on lasagne.layer.GRULayer, with recurrent dropout options.

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden units in the layer.
    resetgate : Gate
        Parameters for the reset gate (:math:`r_t`): :math:`W_{xr}`,
        :math:`W_{hr}`, :math:`b_r`, and :math:`\sigma_r`.
    updategate : Gate
        Parameters for the update gate (:math:`u_t`): :math:`W_{xu}`,
        :math:`W_{hu}`, :math:`b_u`, and :math:`\sigma_u`.
    hidden_update : Gate
        Parameters for the hidden update (:math:`c_t`): :math:`W_{xc}`,
        :math:`W_{hc}`, :math:`b_c`, and :math:`\sigma_c`.
    hid_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial hidden state (:math:`h_0`).
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.
    p : float or scalar tensor
        The probability of recurrent dropout (if p=0, no dropout is applied)
    """

    def __init__(self, incoming, num_units,
                 resetgate=Gate(W_cell=None),
                 updategate=Gate(W_cell=None),
                 hidden_update=Gate(W_cell=None,
                                    nonlinearity=nonlinearities.tanh),
                 hid_init=init.Constant(0.),
                 reset_input=False,
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 p=0.,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have three
        # inputs - the layer input, the mask and the initial hidden state.  We
        # will just provide the layer input as incomings, unless a mask input
        # or initial hidden state was provided.
        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings) - 1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings) - 1

        # Initialize parent layer
        super(GRULayer, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
        self.num_units = num_units
        self.grad_clipping = grad_clipping
        self.reset_input = reset_input
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in all parameters from gates
        (self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')

        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
            hidden_update, 'hidden_update')

        # Initialize hidden state
        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

        # self.r = self.add_param(T.zeros((input_shape[0], input_shape[1], 1), dtype=theano.config.floatX), (input_shape[0], input_shape[1], 1), name='rs')
        # self.z = self.add_param(T.zeros((input_shape[0], input_shape[1], 1), dtype=theano.config.floatX), (input_shape[0], input_shape[1], 1), name='rs')

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable

        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape
        dropout_mask = None
        if not deterministic and self.p:
            one = T.constant(1)
            retain_prob = one - self.p
            dropout_mask_shape = (num_batch, self.num_units)
            dropout_mask = self._srng.binomial(dropout_mask_shape, p=retain_prob, dtype=input.dtype)

        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)

        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0)

        if self.precompute_input:
            # precompute_input inputs*W. W_in is (n_features, 3*num_units).
            # input is then (n_batch, n_time_steps, 3*num_units).
            input = T.dot(input, W_in_stacked)

        # When theano.scan calls step, input_n will be (n_batch, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            s = x[:, n * self.num_units:(n + 1) * self.num_units]
            if self.num_units == 1:
                s = T.addbroadcast(s, 1)  # Theano cannot infer this by itself
            return s

        def slice_b(x, n):
            s = x[n * self.num_units:(n + 1) * self.num_units]
            if self.num_units == 1:
                s = T.addbroadcast(s, 0)  # Theano cannot infer this by itself
            return s

        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        def step(input_n, hid_previous, reset_previous, update_previous, *args):
            hid_previous_dropped = hid_previous
            if not deterministic and self.p:
                hid_previous_dropped = (hid_previous / retain_prob) * dropout_mask

            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            hid_input = T.dot(hid_previous_dropped, W_hid_stacked)

            if self.grad_clipping:
                input_n = theano.gradient.grad_clip(
                    input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

            if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                input_n = T.dot(input_n, W_in_stacked)

            # Reset and update gates
            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0) + slice_b(b_stacked, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1) + slice_b(b_stacked, 1)
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)

            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
            hidden_update_in = slice_w(input_n, 2)
            hidden_update_hid = slice_w(hid_input, 2)
            hidden_update_bias = slice_b(b_stacked, 2)
            if self.reset_input:
                hidden_update = (1 - resetgate) * hidden_update_in + resetgate * hidden_update_hid + hidden_update_bias
            else:
                hidden_update = hidden_update_in + resetgate * hidden_update_hid + hidden_update_bias

            if self.grad_clipping:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hid(hidden_update)

            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid = (1 - updategate) * hid_previous_dropped + updategate * hidden_update
            return hid, resetgate, updategate

        def step_masked(input_n, mask_n, hid_previous, reset_previous, update_previous, *args):
            hid = step(input_n, hid_previous, reset_previous, update_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            hid = T.switch(mask_n, hid, hid_previous)

            return hid

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = [input]
            step_fun = step

        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)

        reset_init = T.zeros((num_batch, 1), dtype=theano.config.floatX)
        update_init = T.zeros((num_batch, 1), dtype=theano.config.floatX)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked, b_stacked]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked]

        if not deterministic and self.p:
            one = T.constant(1)
            retain_prob = one - self.p
            non_seqs += [dropout_mask, retain_prob]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            hid_out, reset_out, update_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init, reset_init, update_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            hid_out, reset_out, update_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init, reset_init, update_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        self.r = reset_out
        self.z = update_out

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out


    def get_gates(self):
        return self.r, self.z


class SGRULayer(MergeLayer):
    """
    GRULayer based on lasagne.layer.GRULayer, with recurrent dropout options.

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden units in the layer.
    resetgate_hidden : Gate
        Parameters for the reset gate (:math:`r_t`): :math:`W_{xr}`,
        :math:`W_{hr}`, :math:`b_r`, and :math:`\sigma_r`.
    resetgate_input : Gate
        Parameters for the reset gate (:math:`i_t`): :math:`W_{xi}`,
        :math:`W_{hi}`, :math:`b_i`, and :math:`\sigma_i`.
    updategate : Gate
        Parameters for the update gate (:math:`u_t`): :math:`W_{xu}`,
        :math:`W_{hu}`, :math:`b_u`, and :math:`\sigma_u`.
    hidden_update : Gate
        Parameters for the hidden update (:math:`c_t`): :math:`W_{xc}`,
        :math:`W_{hc}`, :math:`b_c`, and :math:`\sigma_c`.
    hid_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial hidden state (:math:`h_0`).
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.
    p : float or scalar tensor
        The probability of recurrent dropout (if p=0, no dropout is applied)
    """

    def __init__(self, incoming, num_units,
                 resetgate_hidden=Gate(W_cell=None),
                 resetgate_input=Gate(W_cell=None),
                 updategate=Gate(W_cell=None),
                 hidden_update=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 p=0.,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have three
        # inputs - the layer input, the mask and the initial hidden state.  We
        # will just provide the layer input as incomings, unless a mask input
        # or initial hidden state was provided.
        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings) - 1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings) - 1

        # Initialize parent layer
        super(SGRULayer, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
        self.num_units = num_units
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in all parameters from gates
        (self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate, 'updategate')

        (self.W_in_to_resetgate_hidden, self.W_hid_to_resetgate_hidden, self.b_resetgate_hidden,
         self.nonlinearity_resetgate_hidden) = add_gate_params(resetgate_hidden, 'resetgate_hidden')

        (self.W_in_to_resetgate_input, self.W_hid_to_resetgate_input, self.b_resetgate_input,
         self.nonlinearity_resetgate_input) = add_gate_params(resetgate_input, 'resetgate_input')

        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update, self.b_hidden_update,
         self.nonlinearity_hid) = add_gate_params(hidden_update, 'hidden_update')

        # Initialize hidden state
        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable

        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape
        dropout_mask = None
        if not deterministic and self.p:
            one = T.constant(1)
            retain_prob = one - self.p
            dropout_mask_shape = (num_batch, self.num_units)
            dropout_mask = self._srng.binomial(dropout_mask_shape, p=retain_prob, dtype=input.dtype)

        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate_input, self.W_in_to_resetgate_hidden, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate_input, self.W_hid_to_resetgate_hidden, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)

        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate_input, self.b_resetgate_hidden, self.b_updategate,
             self.b_hidden_update], axis=0)

        if self.precompute_input:
            # precompute_input inputs*W. W_in is (n_features, 4*num_units).
            # input is then (n_batch, n_time_steps, 4*num_units).
            input = T.dot(input, W_in_stacked)

        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            s = x[:, n * self.num_units:(n + 1) * self.num_units]
            if self.num_units == 1:
                s = T.addbroadcast(s, 1)  # Theano cannot infer this by itself
            return s

        def slice_b(x, n):
            s = x[n * self.num_units:(n + 1) * self.num_units]
            if self.num_units == 1:
                s = T.addbroadcast(s, 0)  # Theano cannot infer this by itself
            return s

        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        def step(input_n, hid_previous, *args):
            hid_previous_dropped = hid_previous
            if not deterministic and self.p:
                hid_previous_dropped = (hid_previous / retain_prob) * dropout_mask

            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            hid_input = T.dot(hid_previous_dropped, W_hid_stacked)

            if self.grad_clipping:
                input_n = theano.gradient.grad_clip(
                    input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

            if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                input_n = T.dot(input_n, W_in_stacked)

            # Reset and update gates
            resetgate_input = slice_w(hid_input, 0) + slice_w(input_n, 0) + slice_b(b_stacked, 0)
            resetgate_hidden = slice_w(hid_input, 1) + slice_w(input_n, 1) + slice_b(b_stacked, 1)
            updategate = slice_w(hid_input, 2) + slice_w(input_n, 2) + slice_b(b_stacked, 2)
            resetgate_input = self.nonlinearity_resetgate_input(resetgate_input)
            resetgate_hidden = self.nonlinearity_resetgate_hidden(resetgate_hidden)
            updategate = self.nonlinearity_updategate(updategate)

            # Compute i_t \odot (W_{xc}x_t) + r_t \odot (W_{hc} h_{t - 1})
            hidden_update_in = slice_w(input_n, 3)
            hidden_update_hid = slice_w(hid_input, 3)
            hidden_update_b = slice_b(b_stacked, 3)
            hidden_update = resetgate_input * hidden_update_in + resetgate_hidden * hidden_update_hid + hidden_update_b
            if self.grad_clipping:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hid(hidden_update)

            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid = (1 - updategate) * hid_previous_dropped + updategate * hidden_update
            return hid

        def step_masked(input_n, mask_n, hid_previous, *args):
            hid = step(input_n, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            hid = T.switch(mask_n, hid, hid_previous)

            return hid

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = [input]
            step_fun = step

        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked, b_stacked]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked]

        if not deterministic and self.p:
            one = T.constant(1)
            retain_prob = one - self.p
            non_seqs += [dropout_mask, retain_prob]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out


class MAXRULayer(MergeLayer):
    """
    MAXRULayer based on lasagne.layer.GRULayer, with recurrent dropout options.

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden units in the layer.
    num_time_units : int
        Number of time units in the layer. If None, num_time_units = num_units.
    max_length : int
        Max length of the input sequence.
    P_time : Theano shared variable, numpy array or callable
        Initializer for time matrix (:math:`P`).
    nonlinearity : callable or None
        The nonlinearity that is applied to the time parameters (:math:`P`). If
        None is provided, no nonlinearity will be applied.
    resetgate : Gate
        Parameters for the reset gate (:math:`r_t`): :math:`W_{xr}`,
        :math:`W_{hr}`, :math:`b_r`, and :math:`\sigma_r`.
    updategate : Gate
        Parameters for the update gate (:math:`u_t`): :math:`W_{xu}`,
        :math:`W_{hu}`, :math:`b_u`, and :math:`\sigma_u`.
    hidden_update : Gate
        Parameters for the hidden update (:math:`c_t`): :math:`W_{xc}`,
        :math:`W_{hc}`, :math:`b_c`, and :math:`\sigma_c`.
    time_updategate : Gate
        Parameters for the time update gate (:math:`k_t`): :math:`W_{xk}`,
        :math:`W_{hk}`, :math:`b_k`, and :math:`\sigma_k`.
    time_update : Gate
        Parameters for the time update (:math:`s_t`): :math:`W_{xs}`,
        :math:`W_{hs}`, :math:`b_s`, and :math:`\sigma_s`.
    hid_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial hidden state (:math:`h_0`).
    time_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial time state (:math:`s_0`).
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.
    p : float or scalar tensor
        The probability of recurrent dropout (if p=0, no dropout is applied)
    """

    def __init__(self, incoming, num_units, num_time_units=None, max_length=100,
                 P_time=init.Constant(0.),
                 nonlinearity=nonlinearities.tanh,
                 resetgate=Gate(),
                 updategate=Gate(),
                 hidden_update=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 time_updategate=Gate(W_cell=None),
                 time_update=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 hid_init=init.Constant(0.),
                 time_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 p=0.,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have three
        # inputs - the layer input, the mask and the initial hidden state.  We
        # will just provide the layer input as incomings, unless a mask input
        # or initial hidden state was provided.
        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        self.time_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings) - 1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings) - 1
        if isinstance(time_init, Layer):
            incomings.append(time_init)
            self.time_init_incoming_index = len(incomings) - 1

        # Initialize parent layer
        super(MAXRULayer, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
        self.num_units = num_units
        if num_time_units is None:
            num_time_units = num_units
        self.num_time_units = num_time_units
        self.max_length = max_length
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(input_shape[2:])

        def add_time_gate_params(time_gate, gate_name):
            return (self.add_param(time_gate.W_in, (num_inputs, num_time_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(time_gate.W_hid, (num_units, num_time_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(time_gate.b, (num_time_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    time_gate.nonlinearity)

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in all parameters from gates
        (self.W_in_to_time_updategate, self.W_hid_to_time_updategate, self.b_to_time_updategate,
         self.nonlinearity_time_updategate) = add_time_gate_params(time_updategate,
                                                                   'time_updategate')
        (self.W_in_to_time_update, self.W_hid_to_time_update, self.b_to_time_update,
         self.nonlinearity_time_update) = add_time_gate_params(time_update,
                                                               'time_update')

        (self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate,
                                                         'updategate')
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate, 'resetgate')

        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update, self.b_hidden_update,
         self.nonlinearity_hid) = add_gate_params(hidden_update, 'hidden_update')

        self.W_state_to_updategate = self.add_param(updategate.W_cell, (num_time_units, num_units),
                                                    name="W_state_to_updategate")
        self.W_state_to_resetgate = self.add_param(resetgate.W_cell, (num_time_units, num_units),
                                                   name="W_state_to_resetgate")

        self.P_time = self.add_param(P_time, (max_length, num_time_units), name="P_time")
        # If the provided nonlinearity is None, make it linear
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        # Initialize hidden state
        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)
        # Initialize time state
        if isinstance(time_init, Layer):
            self.time_init = time_init
        else:
            self.time_init = self.add_param(
                time_init, (1, self.num_time_units), name="time_init",
                trainable=learn_init, regularizable=False)

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable

        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with.

        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        time_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.time_init_incoming_index > 0:
            time_init = inputs[self.time_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape
        dropout_mask_hidden = None
        dropout_mask_time = None
        if not deterministic and self.p:
            one = T.constant(1)
            retain_prob = one - self.p
            dropout_mask_shape = (num_batch, self.num_units)
            dropout_mask = self._srng.binomial(dropout_mask_shape, p=retain_prob, dtype=input.dtype)

        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate, self.W_in_to_hidden_update,
             self.W_in_to_time_updategate, self.W_in_to_time_update], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate, self.W_hid_to_hidden_update,
             self.W_hid_to_time_updategate, self.W_hid_to_time_update], axis=1)

        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate, self.b_hidden_update,
             self.b_to_time_updategate, self.b_to_time_update], axis=0)

        W_state_stacked = T.concatenate(
            [self.W_state_to_resetgate, self.W_state_to_updategate], axis=1)

        if self.precompute_input:
            # precompute_input inputs*W. W_in is (n_features, 3*num_units).
            # input is then (n_batch, n_time_steps, 3*num_units).
            input = T.dot(input, W_in_stacked) + b_stacked

        # When theano.scan calls step, input_n will be (n_batch, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            s = x[:, n * self.num_units:(n + 1) * self.num_units]
            if self.num_units == 1:
                s = T.addbroadcast(s, 1)  # Theano cannot infer this by itself
            return s

        def slice_w_time(x, n):
            offset = 3 * self.num_units
            s = x[:, (offset + n * self.num_time_units):(offset + (n + 1) * self.num_time_units)]
            if self.num_time_units == 1:
                s = T.addbroadcast(s, 1)
            return s

        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        def step(p_n, input_n, time_previous, hid_previous, *args):
            hid_previous_dropped = hid_previous
            if not deterministic and self.p:
                hid_previous_dropped = (hid_previous / retain_prob) * dropout_mask

            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            hid_input = T.dot(hid_previous_dropped, W_hid_stacked)

            if self.grad_clipping:
                input_n = theano.gradient.grad_clip(
                    input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

            if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # time update gates
            time_updategate = slice_w_time(hid_input, 0) + slice_w_time(input_n, 0)
            time_update = slice_w_time(hid_input, 1) + slice_w_time(input_n, 1)
            time_updategate = self.nonlinearity_time_updategate(time_updategate)
            time_update = self.nonlinearity_time_update(time_update)

            # compute time_state
            time = (1 - time_updategate) * time_previous + time_updategate * time_update
            time = time * (self.nonlinearity(p_n))

            time_input = T.dot(time, W_state_stacked)

            if self.grad_clipping:
                time_input = theano.gradient.grad_clip(
                    time_input, -self.grad_clipping, self.grad_clipping)

            # Reset and update gates
            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0) + slice_w(time_input, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1) + slice_w(time_input, 1)
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)

            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
            hidden_update_in = slice_w(input_n, 2)
            hidden_update_hid = slice_w(hid_input, 2)
            hidden_update = hidden_update_in + resetgate * hidden_update_hid
            if self.grad_clipping:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hid(hidden_update)

            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid = (1 - updategate) * hid_previous_dropped + updategate * hidden_update
            return [time, hid]

        def step_masked(p_n, input_n, mask_n, time_previous, hid_previous, *args):
            time, hid = step(p_n, input_n, time_previous, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            time = T.switch(mask_n, time, time_previous)
            hid = T.switch(mask_n, hid, hid_previous)

            return [time, hid]

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [self.P_time[:seq_len], input, mask]
            step_fun = step_masked
        else:
            sequences = [self.P_time[:seq_len], input]
            step_fun = step

        if not isinstance(self.time_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            time_init = T.dot(T.ones((num_batch, 1)), self.time_init)

        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked, W_state_stacked]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        if not deterministic and self.p:
            one = T.constant(1)
            retain_prob = one - self.p
            non_seqs += [dropout_mask, retain_prob]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            time_out, hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[time_init, hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            time_out, hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[time_init, hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out
