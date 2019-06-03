"""
    Model Components
"""
import sys
import traceback
import numpy as np
from threading import Thread
import os
import multiprocessing
import math
from collections import OrderedDict

from paddle import fluid
from paddle.fluid import layers
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.layer_helper import LayerHelper 
from paddle.fluid.initializer import Constant

from utils import TracebackWrapper, save_pickle, seq_len_2_lod, tik, tok

def fluid_expand_dims(tensor, dim):
    """
    like:
    # 't' is a tensor of shape [2]
    tf.shape(tf.expand_dims(t, 0))  # [1, 2]
    tf.shape(tf.expand_dims(t, 1))  # [2, 1]
    tf.shape(tf.expand_dims(t, -1))  # [2, 1]

    # 't2' is a tensor of shape [2, 3, 5]
    tf.shape(tf.expand_dims(t2, 0))  # [1, 2, 3, 5]
    tf.shape(tf.expand_dims(t2, 2))  # [2, 3, 1, 5]
    tf.shape(tf.expand_dims(t2, 3))  # [2, 3, 5, 1]

    """
    shape = list(tensor.shape)
    assert dim <= len(shape) and dim >= -len(shape)-1, 'dim %d exceed shape %s' % (dim, shape)
    if dim < 0:
        dim = dim + 1 + len(shape)
    shape.insert(dim, 1)
    new_tensor = layers.reshape(tensor, shape=shape)
    return new_tensor

def fluid_fill_constant_like(tensor, value):
    zeros = tensor * 0
    return zeros + value

def scatter(input, index, updates, name=None):
    """
    **Scatter Layer**
    by Lihang Liu. 
    There's a bug in Python API scatter for parameter checking,
    please refer to (https://github.com/PaddlePaddle/Paddle/issues/12725).
    Output is obtained by updating the input on selected indices on the first
    axis.
    .. math::
        Out = X
        Out[Ids] = Updates
    Args:
        input (Variable): The source input with rank>=1.
        index (Variable): The index input with rank=1. Its dtype should be
                          int32 or int64 as it is used as indexes.
        updates (Variable): The updated value of scatter op.
        name (str|None): The output variable name. Default None.
    Returns:
        output (Variable): The output is a tensor with the same shape as input.
    Examples:
        .. code-block:: python
            output = fluid.layers.scatter(input, index, updates)
    """
    helper = LayerHelper('scatter', **locals())
    dtype = helper.input_dtype()
    out = helper.create_tmp_variable(dtype)
    helper.append_op(
        type="scatter",
        inputs={"X": input,
                "Ids": index,
                "Updates": updates},
        outputs={"Out": out})
    return out

def fluid_batch_norm(input,
               act=None,
               is_test=False,
               momentum=0.9,
               epsilon=1e-05,
               param_attr=None,
               bias_attr=None,
               mean_attr=None,
               var_attr=None,
               data_layout='NCHW',
               in_place=False,
               name=None,
               moving_mean_name=None,
               moving_variance_name=None,
               do_model_average_for_mean_and_var=False,
               fuse_with_relu=False):
    """
    **Batch Normalization Layer**
    Editted by Lihang Liu for the reason of exposing mean_attr and var_attr.

    Can be used as a normalizer function for conv2d and fully_connected operations.
    The required data format for this layer is one of the following:

    1. NHWC `[batch, in_height, in_width, in_channels]`

    2. NCHW `[batch, in_channels, in_height, in_width]`

    Refer to `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/pdf/1502.03167.pdf>`_
    for more details.

    :math:`input` is the input features over a mini-batch.

    ..  math::

        \\mu_{\\beta} &\\gets \\frac{1}{m} \\sum_{i=1}^{m} x_i \\qquad &//\\
        \ mini-batch\ mean \\\\
        \\sigma_{\\beta}^{2} &\\gets \\frac{1}{m} \\sum_{i=1}^{m}(x_i - \\
        \\mu_{\\beta})^2 \\qquad &//\ mini-batch\ variance \\\\
        \\hat{x_i} &\\gets \\frac{x_i - \\mu_\\beta} {\\sqrt{\\
        \\sigma_{\\beta}^{2} + \\epsilon}} \\qquad &//\ normalize \\\\
        y_i &\\gets \\gamma \\hat{x_i} + \\beta \\qquad &//\ scale\ and\ shift

    Args:
        input(variable): The input variable which is a LoDTensor.
        act(string, Default None): Activation type, linear|relu|prelu|...
        is_test(bool, Default False): Used for training or training.
        momentum(float, Default 0.9):
        epsilon(float, Default 1e-05):
        param_attr(ParamAttr|None): The parameter attribute for Parameter `scale`
             of batch_norm. If it is set to None or one attribute of ParamAttr, batch_norm
             will create ParamAttr as param_attr. If the Initializer of the param_attr
             is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr(ParamAttr|None): The parameter attribute for the bias of batch_norm.
             If it is set to None or one attribute of ParamAttr, batch_norm
             will create ParamAttr as bias_attr. If the Initializer of the bias_attr
             is not set, the bias is initialized zero. Default: None.
        data_layout(string, default NCHW): NCHW|NHWC
        in_place(bool, Default False): Make the input and output of batch norm reuse memory.
        name(string, Default None): A name for this layer(optional). If set None, the layer
            will be named automatically.
        moving_mean_name(string, Default None): The name of moving_mean which store the global Mean.
        moving_variance_name(string, Default None): The name of the moving_variance which store the global Variance.
        do_model_average_for_mean_and_var(bool, Default False): Do model average for mean and variance or not.
        fuse_with_relu (bool): if True, this OP performs relu after batch norm.

    Returns:
        Variable: A tensor variable which is the result after applying batch normalization on the input.

    Examples:

        .. code-block:: python

            hidden1 = fluid.layers.fc(input=x, size=200, param_attr='fc1.w')
            hidden2 = fluid.layers.batch_norm(input=hidden1)
    """
    assert bias_attr is not False, "bias_attr should not be False in batch_norm."
    helper = LayerHelper('batch_norm', **locals())
    dtype = helper.input_dtype()

    input_shape = input.shape
    if data_layout == 'NCHW':
        channel_num = input_shape[1]
    else:
        if data_layout == 'NHWC':
            channel_num = input_shape[-1]
        else:
            raise ValueError("unsupported data layout:" + data_layout)

    param_shape = [channel_num]

    # create parameter
    scale = helper.create_parameter(
        attr=helper.param_attr,
        shape=param_shape,
        dtype=dtype,
        default_initializer=Constant(1.0))

    bias = helper.create_parameter(
        attr=helper.bias_attr, shape=param_shape, dtype=dtype, is_bias=True)

    if mean_attr is None:
        mean = helper.create_parameter(
            attr=ParamAttr(
                name=moving_mean_name,
                initializer=Constant(0.0),
                trainable=False,
                do_model_average=do_model_average_for_mean_and_var),
            shape=param_shape,
            dtype=input.dtype)
    else:
        mean = helper.create_parameter(
            attr=mean_attr,
            shape=param_shape,
            dtype=input.dtype)
    mean.stop_gradient = True

    if var_attr is None:
        variance = helper.create_parameter(
            attr=ParamAttr(
                name=moving_variance_name,
                initializer=Constant(1.0),
                trainable=False,
                do_model_average=do_model_average_for_mean_and_var),
            shape=param_shape,
            dtype=input.dtype)
    else:
        variance = helper.create_parameter(
            attr=var_attr,
            shape=param_shape,
            dtype=input.dtype)
    variance.stop_gradient = True

    # create output
    # mean and mean_out share the same memory
    mean_out = mean
    # variance and variance out share the same memory
    variance_out = variance
    saved_mean = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)
    saved_variance = helper.create_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)

    batch_norm_out = input if in_place else helper.create_variable_for_type_inference(
        dtype)

    helper.append_op(
        type="batch_norm",
        inputs={
            "X": input,
            "Scale": scale,
            "Bias": bias,
            "Mean": mean,
            "Variance": variance
        },
        outputs={
            "Y": batch_norm_out,
            "MeanOut": mean_out,
            "VarianceOut": variance_out,
            "SavedMean": saved_mean,
            "SavedVariance": saved_variance
        },
        attrs={
            "momentum": momentum,
            "epsilon": epsilon,
            "is_test": is_test,
            "use_mkldnn": False,
            "fuse_with_relu": fuse_with_relu
        })

    return helper.append_activation(batch_norm_out)


# def fluid_sequence_scatter(input, index, offset, updates):
#     """
#     args:
#         input: 1-level LoDTensor, 'float32' only
#         index: 1-d tensor of the sequence index, 'int32' only
#         offset: the same shape and dtype as index
#         updates: (len(index), input[1:])
#     return:
#         output = input
#         output[index + offset] = updates
#         lod_set(output, input)
#     """
#     # assert input.lod_level == 1, input
#     assert index.shape == offset.shape
#     assert input.shape[1:] == updates.shape[1:]
#     new_index = index + offset
#     new_index.stop_gradient = True
#     output = layers.scatter(input, new_index, updates)
#     return layers.lod_reset(output, input)


def fluid_sequence_scatter(input, index, value):
    """
    args:
        input: 1-level LoDTensor
        index: 1-d tensor of the sequence index
        value: scalar
    return:
        output = input
        output[index + offset] = updates
        lod_set(output, input)
    """
    offset = fluid_sequence_get_offset(input)
    offset_index = index + offset
    offset_index.stop_gradient = True
    updates = fluid.layers.fill_constant_batch_size_like(input, shape=input.shape, value=value, dtype=input.dtype)
    output = layers.scatter(input, layers.cast(offset_index, 'int32'), updates)
    return layers.lod_reset(output, input)


def fluid_sequence_get_offset(input):
    seq_len = fluid_sequence_get_seq_len(input)
    offset = fluid_get_offset(layers.reshape(seq_len, [-1]))
    return offset

def fluid_get_offset(seq_len):
    """
    args:
        seq_len: (-1)
    return:
        offset: the same shape as seq_len,
            cumsum(seq_len) - seq_len 
    """
    assert len(seq_len.shape) == 1
    csum = layers.cumsum(layers.cast(seq_len, 'float32'), exclusive=True)
    return layers.cast(csum, 'int64')


def fluid_sequence_delay(input, OOV):
    """
    args:
        input.data = [1,2,3, 4,5]
        input.lod = [[0, 3, 5]]
    return:
        output.data = [2,3,0, 5,0]
        output.lod = [[0, 3, 5]]
    """
    seq_len = fluid_sequence_get_seq_len(input)
    zeros = layers.fill_constant_batch_size_like(seq_len, shape=[-1,1], value=0, dtype='int64')
    ones = layers.fill_constant_batch_size_like(seq_len, shape=[-1,1], value=1, dtype='int64')
    oov = layers.sequence_slice(input, zeros, ones) * 0 + OOV
    oov.stop_gradient = True
    input_padded = layers.sequence_concat([input, oov])
    output = layers.sequence_slice(input_padded, ones, seq_len)
    return output


def fluid_sequence_advance(input, OOV):
    """
    args:
        input.data = [1,2,3, 4,5]
        input.lod = [[0, 3, 5]]
    return:
        output.data = [0,1,2, 0,4]
        output.lod = [[0, 3, 5]]
    """
    seq_len = fluid_sequence_get_seq_len(input)
    zeros = layers.fill_constant_batch_size_like(seq_len, shape=[-1,1], value=0, dtype='int64')
    ones = layers.fill_constant_batch_size_like(seq_len, shape=[-1,1], value=1, dtype='int64')
    oov = layers.sequence_slice(input, zeros, ones) * 0 + OOV
    oov.stop_gradient = True
    input_padded = layers.sequence_concat([oov, input])
    output = layers.sequence_slice(input_padded, zeros, seq_len)
    return output


def fluid_sequence_pad(input, pad_value, maxlen=None):
    """
    args:
        input: (batch*seq_len, dim)
    returns:
        (batch, max_seq_len, dim)
    """
    pad_value = layers.cast(fluid.layers.assign(input=np.array([pad_value], 'float32')), input.dtype)
    input_padded, _ = layers.sequence_pad(input, pad_value, maxlen=maxlen)    # (batch, max_seq_len, 1), (batch, 1)
                                                                                # TODO, maxlen=300, used to solve issues: https://github.com/PaddlePaddle/Paddle/issues/14164
    return input_padded


def fluid_split(input, num_or_sections, dim=-1, name=None):
    if num_or_sections == 1:
        return [input]
    else:
        return layers.split(input, num_or_sections, dim, name)


def fluid_index1d(input, dim0, keep_dim=False):
    x0 = layers.gather(input, layers.fill_constant(shape=[1], value=dim0, dtype='int32'))
    if not keep_dim:
        x0 = layers.squeeze(x0, axes=[0])
    return x0


def fluid_index2d(input, dim0, dim1, keep_dim=False):
    return fluid_index1d(
                fluid_index1d(input, dim0, keep_dim=keep_dim), 
                dim1, 
                keep_dim=keep_dim)


def fluid_create_lod_tensor(array, lod, place):
    assert isinstance(array, np.ndarray), (type(array))
    tensor = fluid.LoDTensor()
    tensor.set(array, place)
    tensor.set_lod(lod)
    return tensor


def fluid_sequence_get_pos(lodtensor):
    """
    args:
        lodtensor: lod = [[0,4,7]]
    return:
        pos: lod = [[0,4,7]]
             data = [0,1,2,3,0,1,3]
             shape = [-1, 1]
    """
    lodtensor = layers.reduce_sum(lodtensor, dim=1, keep_dim=True) 
    assert lodtensor.shape == (-1, 1), (lodtensor.shape())
    ones = layers.cast(lodtensor * 0 + 1, 'float32')        # (batch*seq_len, 1)
    ones_padded = fluid_sequence_pad(ones, 0)               # (batch, max_seq_len, 1)
    ones_padded = layers.squeeze(ones_padded, [2])          # (batch, max_seq_len)
    seq_len = layers.cast(layers.reduce_sum(ones_padded, 1, keep_dim=True), 'int64')    # (batch, 1)
    pos = layers.cast(layers.cumsum(ones_padded, 1, exclusive=True), 'int64')
    pos = layers.sequence_unpad(pos, seq_len)               # (batch*seq_len, 1)
    pos.stop_gradient = True
    return pos


# def fluid_sequence_get_seq_len(lodtensor):
#     """
#     args:
#         lodtensor: lod = [[0,4,7]]
#     return:
#         seq_len: lod = []
#              data = [4, 3]
#              shape = [-1, 1]
#     """
#     lodtensor = layers.slice(lodtensor, axes=[1], starts=[0], ends=[1])
#     assert lodtensor.shape == (-1, 1), (lodtensor.shape())
#     ones = layers.cast(lodtensor * 0 + 1, 'float32')        # (batch*seq_len, 1)
#     ones_padded = fluid_sequence_pad(ones, 0)               # (batch, max_seq_len, 1)
#     ones_padded = layers.squeeze(ones_padded, [2])          # (batch, max_seq_len)
#     seq_len = layers.cast(layers.reduce_sum(ones_padded, 1, keep_dim=True), 'int64')    # (batch, 1)
#     return seq_len


def fluid_sequence_get_seq_len(input):
    """
    args:
        input: lod = [[0,4,7]]
    return:
        seq_len: lod = []
             data = [4, 3]
             shape = [-1, 1]
    """
    zero = layers.cast(fluid.layers.assign(input=np.array([0], 'float32')), input.dtype)
    input_padded, seq_len = layers.sequence_pad(input, zero)
    return seq_len


def fluid_sequence_first_step(lodtensor):
    """
    return a lod tensor
    """
    offset = layers.fill_constant_batch_size_like(lodtensor, shape=[-1,1], value=0, dtype='int64')
    length = layers.fill_constant_batch_size_like(lodtensor, shape=[-1,1], value=1, dtype='int64')
    res = layers.sequence_slice(lodtensor, offset=offset, length=length)
    return res


def fluid_sequence_index(input, index):
    """
    index: (batch_size, 1)
    """
    ones = layers.fill_constant_batch_size_like(input, shape=[-1,1], value=1, dtype='int64')
    output = layers.sequence_slice(input, offset=index, length=ones)
    return output


def get_num_devices(use_cuda):
    if use_cuda:
        return fluid.core.get_cuda_device_count()
    else:
        return int(os.environ.get('CPU_NUM', 
                    multiprocessing.cpu_count()))


def concat_list_array(list_array, dtype, need_padding=False):
    """
    no padding version and no expanding

    args:
        list_array: 
            [(seq[0], *), (seq[1], *), ...]
    """
    concated_array = []
    list_seq_len = []
    for array in list_array:
        l = array.tolist()
        for x in l:
            if isinstance(x, list) and len(x) == 0:
                x.append(0)
        concated_array += l
        list_seq_len.append(len(array))

    if need_padding:
        concated_array = VariableSizeArray(concated_array).align(0).astype(dtype)
    else:
        concated_array = TracebackWrapper(np.array)(concated_array, dtype)

    lod = [seq_len_2_lod(list_seq_len)]
    return concated_array, lod


def executor_run_with_fetch_dict(exe, program=None, feed=None, fetch_dict=None, 
                                feed_var_name='feed', fetch_var_name='fetch', 
                                scope=None, return_numpy=True, use_program_cache=False):
    """
    args:
        use fetch_dict to replace fetch_list
    return:
        res_fetch_dict: a dict to replace a list
    """
    fetch_names = fetch_dict.keys()
    fetch_list = fetch_dict.values()
    res_fetch_list = exe.run(program=program, feed=feed, fetch_list=fetch_list, 
                            feed_var_name=feed_var_name, fetch_var_name=fetch_var_name, 
                            scope=scope, return_numpy=return_numpy, use_program_cache=use_program_cache)
    res_fetch_dict = OrderedDict()
    for name, res in zip(fetch_names, res_fetch_list):
        res_fetch_dict[name] = res
    return res_fetch_dict


def parallel_executor_run_with_fetch_dict(exe, fetch_dict, feed=None, feed_dict=None, return_numpy=True):
    """
    args:
        use fetch_dict to replace fetch_list
    return:
        res_fetch_dict: a dict to replace a list
    """
    fetch_names = fetch_dict.keys()
    fetch_list = fetch_dict.values()
    res_fetch_list = exe.run(fetch_list=[x.name for x in fetch_list], 
                             feed=feed, feed_dict=feed_dict, return_numpy=return_numpy)
    res_fetch_dict = OrderedDict()
    for name, res in zip(fetch_names, res_fetch_list):
        res_fetch_dict[name] = res
    return res_fetch_dict



