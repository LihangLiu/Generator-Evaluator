#!/usr/bin/env python
# coding=utf8
import numpy as np
import sys
import os
from os.path import exists
import copy
import logging
from collections import OrderedDict

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr

PARL_DIR = os.environ['PARL_DIR']
sys.path.append(PARL_DIR)

import parl.layers as layers
from parl.framework.algorithm import Model

from fluid_utils import (fluid_batch_norm, fluid_sequence_pad, fluid_sequence_get_pos, 
                        fluid_sequence_index, fluid_sequence_scatter, fluid_sequence_get_seq_len)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # filename='hot_rl.log', 


from parl.layers.layer_wrappers import update_attr_name, LayerFunc, check_caller_name
def batch_norm(data_layout='NCHW',
       param_attr=None,
       bias_attr=None,
       mean_attr=None,
       var_attr=None,
       moving_mean_name=None,
       moving_variance_name=None,
       do_model_average_for_mean_and_var=False,
       name=None):
    """
    simplified, wait for PARL upgrade
    """
    default_name = "batch_norm"
    param_attr = update_attr_name(name, default_name, param_attr, False)
    bias_attr = update_attr_name(name, default_name, bias_attr, True)
    mean_attr = update_attr_name(name, default_name, mean_attr, False)
    var_attr = update_attr_name(name, default_name, var_attr, False)
    check_caller_name()

    class batch_norm_(LayerFunc):
        def __init__(self):
            super(batch_norm_, self).__init__(param_attr, bias_attr)

        def __call__(self, input, is_test):
            # return fluid.layers.batch_norm(input=input,
            #                              is_test=is_test,
            #                              data_layout=data_layout,
            #                              param_attr=self.param_attr,
            #                              bias_attr=self.bias_attr,
            #                              moving_mean_name=moving_mean_name,
            #                              moving_variance_name=moving_variance_name,
            #                              do_model_average_for_mean_and_var=do_model_average_for_mean_and_var)
            return fluid_batch_norm(input=input,
                                     is_test=is_test,
                                     # momentum=0.0,
                                     data_layout=data_layout,
                                     param_attr=self.param_attr,
                                     bias_attr=self.bias_attr,
                                     mean_attr=mean_attr,
                                     var_attr=var_attr,
                                     moving_mean_name=moving_mean_name,
                                     moving_variance_name=moving_variance_name,
                                     do_model_average_for_mean_and_var=do_model_average_for_mean_and_var)

    return batch_norm_()


def default_normal_initializer(nf=128):
    return fluid.initializer.TruncatedNormal(loc=0.0, scale=np.sqrt(1.0/nf))


def default_param_clip():
    return fluid.clip.GradientClipByValue(1.0)


def default_regularizer():
    return None
    # return fluid.regularizer.L2Decay(0.01)


def default_batch_norm(name=None):
    return batch_norm(data_layout='NHWC',
                   param_attr=ParamAttr(initializer=default_normal_initializer(),
                                        gradient_clip=default_param_clip()),
                   bias_attr=ParamAttr(initializer=fluid.initializer.Constant(value=0.0),
                                        gradient_clip=default_param_clip()),
                   mean_attr=ParamAttr(initializer=fluid.initializer.Constant(value=0.0),
                                        trainable=False,
                                        do_model_average=False),
                   var_attr=ParamAttr(initializer=fluid.initializer.Constant(value=1.0),
                                        trainable=False,
                                        do_model_average=False),
                   name=name)

def default_fc(size, num_flatten_dims=1, act=None, name=None):
    return layers.fc(size=size,
                   num_flatten_dims=num_flatten_dims,
                   param_attr=ParamAttr(initializer=default_normal_initializer(size),
                                        gradient_clip=default_param_clip(),
                                        regularizer=default_regularizer()),
                   bias_attr=ParamAttr(initializer=fluid.initializer.Constant(value=0.0),
                                        gradient_clip=default_param_clip(),
                                        regularizer=default_regularizer()),
                   act=act,
                   name=name)

def default_embedding(size, name):
    gradient_clip = default_param_clip()
    reg = fluid.regularizer.L2Decay(1e-5)   # IMPORTANT, to prevent overfitting.
    embed = layers.embedding(name=name,
                            size=size,
                            param_attr=ParamAttr(initializer=fluid.initializer.Xavier(),
                                                gradient_clip=gradient_clip,
                                                regularizer=reg),
                            is_sparse=False)
    return embed

def default_drnn(nf, is_reverse=False, name=None):
    return layers.dynamic_gru(size=nf,
                            param_attr=ParamAttr(initializer=default_normal_initializer(nf),
                                                gradient_clip=default_param_clip(),
                                                regularizer=default_regularizer()),
                            bias_attr=ParamAttr(initializer=fluid.initializer.Constant(value=0.0),
                                                gradient_clip=default_param_clip(),
                                                regularizer=default_regularizer()),
                            is_reverse=is_reverse,
                            name=name)


class BaseModel(Model):
    """
    Parent model for generator-evaluator models
    """
    def __init__(self, conf, npz_config):
        super(BaseModel, self).__init__()
        self.npz_config = npz_config
        self.data_attributes = conf.data_attributes
        # feature related initialization
        self.item_slot_names = conf.item_slot_names
        self.user_slot_names = conf.user_slot_names

        self.hidden_size = 128
        self.embed_size = 16
        self.SAFE_EPS = 1e-6
        self.BIG_VALUE = 1e6

    def get_input_specs(self):
        """ignore"""
        return []

    def get_action_specs(self):
        """ignore"""
        return []

    def _create_params(self):
        raise NotImplementedError()

    def _build_embeddings(self, inputs, list_names):
        list_embed = []
        for name in list_names:
            # message = "%s %d" % (name, self.npz_config['embedding_size'][name])
            # layers.Print(layers.reduce_max(inputs[name]), summarize=32, print_tensor_lod=False, message=message)
            c_embed = self.dict_data_embed_op[name](inputs[name])
            list_embed.append(c_embed)                              # (batch*seq_lens, 16)
        concated_embed = layers.concat(input=list_embed, axis=1)    # (batch*seq_lens, concat_dim)
        concated_embed = layers.softsign(concated_embed)
        return concated_embed

    def create_inputs(self, mode):
        raise NotImplementedError()

    def forward(self, inputs, mode):
        raise NotImplementedError()

    ### sampling functions ###

    def _cut_by_decode_len(self, input, decode_len):
        zeros = layers.fill_constant_batch_size_like(input, shape=[-1,1], value=0, dtype='int64')
        output = layers.sequence_slice(layers.cast(input, 'float32'), offset=zeros, length=decode_len)
        return layers.cast(output, input.dtype)

    def eps_greedy_sampling(self, scores, mask, eps):
        scores = scores * mask
        scores_padded = layers.squeeze(fluid_sequence_pad(scores, 0, maxlen=128), [2])  # (b*s, 1) -> (b, s, 1) -> (b, s)
        mask_padded = layers.squeeze(fluid_sequence_pad(mask, 0, maxlen=128), [2])
        seq_lens = fluid_sequence_get_seq_len(scores)

        def get_greedy_prob(scores_padded, mask_padded):
            s = scores_padded - (mask_padded*(-1) + 1) * self.BIG_VALUE
            max_value = layers.reduce_max(s, dim=1, keep_dim=True)
            greedy_prob = layers.cast(s >= max_value, 'float32')
            return greedy_prob
        greedy_prob = get_greedy_prob(scores_padded, mask_padded)
        eps_prob = mask_padded * eps / layers.reduce_sum(mask_padded, dim=1, keep_dim=True)

        final_prob = (greedy_prob + eps_prob) * mask_padded
        final_prob = final_prob / layers.reduce_sum(final_prob, dim=1, keep_dim=True)

        sampled_id = layers.reshape(layers.sampling_id(final_prob), [-1, 1])
        max_id = layers.cast(layers.cast(seq_lens, 'float32') - 1, 'int64')
        sampled_id = layers.elementwise_min(sampled_id, max_id)
        return layers.cast(sampled_id, 'int64')

    def softmax_sampling(self, scores, mask, eta):
        scores = scores * mask
        scores_padded = layers.squeeze(fluid_sequence_pad(scores, 0, maxlen=128), [2])  # (b*s, 1) -> (b, s, 1) -> (b, s)
        mask_padded = layers.squeeze(fluid_sequence_pad(mask, 0, maxlen=128), [2])
        seq_lens = fluid_sequence_get_seq_len(scores)

        def normalize(scores_padded, mask_padded):
            mean_S = layers.reduce_sum(scores_padded, dim=1, keep_dim=True) / layers.reduce_sum(mask_padded, dim=1, keep_dim=True)
            S = scores_padded - mean_S
            std_S = layers.sqrt(layers.reduce_sum(layers.square(S * mask_padded), dim=1, keep_dim=True))
            return S / (std_S + self.SAFE_EPS)
        
        norm_S = normalize(scores_padded, mask_padded)
        # set mask to large negative values
        norm_S = norm_S * mask_padded - (mask_padded*(-1) + 1) * self.BIG_VALUE
        soft_prob = layers.softmax(norm_S / eta) * mask_padded
        sampled_id = layers.reshape(layers.sampling_id(soft_prob), [-1, 1])
        max_id = layers.cast(layers.cast(seq_lens, 'float32') - 1, 'int64')
        sampled_id = layers.elementwise_min(sampled_id, max_id)
        return layers.cast(sampled_id, 'int64')

    def sampling_rnn_forward(self, independent_item_fc, independent_hidden, independent_pos_embed):
        raise NotImplementedError()

        # example:
        gru_input = self.item_gru_fc_op(layers.concat([independent_item_fc, independent_pos_embed], 1))
        next_hidden = self.item_gru_op(gru_input, independent_hidden)
        scores = self.out_Q_fc2_op(self.out_Q_fc1_op(next_hidden))
        return next_hidden, scores

    def sampling_rnn(self, item_fc, h_0, pos_embed, forward_func, sampling_type, eps=0, eta=1):
        mask = layers.reduce_sum(item_fc, dim=1, keep_dim=True) * 0 + 1
        drnn = fluid.layers.DynamicRNN()
        with drnn.block():
            # e.g. batch_size = 2
            _ = drnn.step_input(item_fc)
            cur_pos_embed = drnn.step_input(pos_embed)          # lod = []
            cur_h_0 = drnn.memory(init=h_0, need_reorder=True)  # lod = [0,1,2]
            item_fc = drnn.static_input(item_fc)
            mask = drnn.memory(init=mask, need_reorder=True)

            # step_input will remove lod info
            cur_pos_embed = layers.lod_reset(cur_pos_embed, cur_h_0)

            # expand
            expand_h_0 = layers.sequence_expand(cur_h_0, item_fc)               # lod = [0,1,2,3,4,5,6,7]
            expand_pos_embed = layers.sequence_expand(cur_pos_embed, item_fc)   # lod = [0,1,2,3,4,5,6,7]
            expand_item_fc = layers.lod_reset(item_fc, expand_h_0)
            # forward
            expand_next_h_0, expand_scores = forward_func(expand_item_fc, expand_h_0, expand_pos_embed)
            # reset result lod
            expand_next_h_0 = layers.lod_reset(expand_next_h_0, item_fc)        # lod = [0,4,7]
            expand_scores = layers.lod_reset(expand_scores, item_fc)            # lod = [0,4,7]

            if sampling_type == 'eps_greedy':
                selected_index = self.eps_greedy_sampling(expand_scores, mask, eps=eps)
            elif sampling_type == 'softmax':
                selected_index = self.softmax_sampling(expand_scores, mask, eta=eta)

            drnn.output(selected_index)

            next_h_0 = fluid_sequence_index(expand_next_h_0, selected_index)
            next_mask = fluid_sequence_scatter(mask, layers.reshape(selected_index, [-1]), 0.0)

            # update
            drnn.update_memory(cur_h_0, next_h_0)
            drnn.update_memory(mask, next_mask)

        drnn_output = drnn()
        return drnn_output


