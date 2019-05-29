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

from fluid_utils import fluid_batch_norm

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
            c_embed = self.dict_data_embed_op[name](inputs[name])
            list_embed.append(c_embed)                              # (batch*seq_lens, 16)
        concated_embed = layers.concat(input=list_embed, axis=1)    # (batch*seq_lens, concat_dim)
        concated_embed = layers.softsign(concated_embed)
        return concated_embed

    def create_inputs(self, mode):
        raise NotImplementedError()

    def forward(self, inputs, mode):
        raise NotImplementedError()


