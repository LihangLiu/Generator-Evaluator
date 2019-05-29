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

from fluid_utils import fluid_sequence_get_pos, fluid_sequence_first_step
from base_net import BaseModel, default_fc, default_batch_norm, default_embedding, default_drnn

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # filename='hot_rl.log', 


class DNN(BaseModel):
    """
    """
    def __init__(self, conf, npz_config):
        super(DNN, self).__init__(conf, npz_config)
        self._create_params()

    def _create_params(self):
        ### embed
        self.dict_data_embed_op = {}
        list_names = self.item_slot_names + self.user_slot_names + ['pos']
        for name in list_names:
            vob_size = self.npz_config['embedding_size'][name] + 1
            self.dict_data_embed_op[name] = default_embedding([vob_size, self.embed_size], 'embed_' + name)

        self.user_feature_fc_op = default_fc(self.hidden_size, act='relu', name='user_feature_fc')

        self.item_fc_op = default_fc(self.hidden_size * 3, act='relu', name='item_fc')

        self.out_fc1_op = default_fc(self.hidden_size, act='relu', name='out_fc1')
        self.out_fc2_op = default_fc(2, act='softmax', name='out_fc2')

    def create_inputs(self, mode):
        """create layers.data here"""
        inputs = OrderedDict()
        data_attributes = copy.deepcopy(self.data_attributes)
        data_attributes['pos'] = {'shape': (-1, 1), 'dtype': 'int64', 'lod_level': 1}
        data_attributes['click_id'] = {'shape': (-1, 1), 'dtype': 'int64', 'lod_level': 1}
        data_attributes['prev_hidden'] = {'shape': (-1, self.hidden_size), 'dtype': 'float32', 'lod_level': 1}

        if mode in ['train', 'test']:
            list_names = self.item_slot_names + self.user_slot_names + ['pos', 'click_id']
        elif mode in ['inference']:
            list_names = self.item_slot_names + self.user_slot_names + ['pos']
        elif mode == 'infer_init':
            list_names = self.user_slot_names
        elif mode == 'infer_onestep':
            list_names = self.item_slot_names + ['pos', 'prev_hidden']
        else:
            raise NotImplementedError(mode)

        for name in list_names:
            p = data_attributes[name]
            inputs[name] = layers.data(name=name, shape=p['shape'], dtype=p['dtype'], lod_level=p['lod_level'])
        return inputs

    def user_encode(self, inputs):
        user_embedding = self._build_embeddings(inputs, self.user_slot_names)
        user_feature = self.user_feature_fc_op(user_embedding)
        return user_feature

    def item_decode(self, inputs, user_feature):
        item_embedding = self._build_embeddings(inputs, self.item_slot_names)
        item_fc = self.item_fc_op(item_embedding)
        concat_fc = layers.concat([item_fc, layers.sequence_expand_as(user_feature, item_fc)], 1)
        return concat_fc

    ### main functions ###

    def forward(self, inputs):
        """forward"""
        user_feature = self.user_encode(inputs)
        concat_fc = self.item_decode(inputs, user_feature)
        click_prob = self.out_fc2_op(self.out_fc1_op(concat_fc))
        return click_prob

    def infer_init(self, inputs):
        """inference only the init part"""
        user_feature = self.user_encode(inputs)
        return user_feature

    def infer_onestep(self, inputs):
        """inference the gru-unit by one step"""
        user_feature = inputs['prev_hidden']
        concat_fc = self.item_decode(inputs, user_feature)
        click_prob = self.out_fc2_op(self.out_fc1_op(concat_fc))
        return user_feature, click_prob


class UniRNN(BaseModel):
    """
    """
    def __init__(self, conf, npz_config):
        super(UniRNN, self).__init__(conf, npz_config)
        self._create_params()

    def _create_params(self):
        ### embed
        self.dict_data_embed_op = {}
        list_names = self.item_slot_names + self.user_slot_names
        for name in list_names:
            vob_size = self.npz_config['embedding_size'][name] + 1
            self.dict_data_embed_op[name] = default_embedding([vob_size, self.embed_size], 'embed_' + name)

        self.user_feature_fc_op = default_fc(self.hidden_size, act='relu', name='user_feature_fc')

        self.item_fc_op = default_fc(self.hidden_size * 3, act='relu', name='item_fc')
        self.item_gru_op = default_drnn(self.hidden_size, name='item_gru')

        self.out_fc1_op = default_fc(self.hidden_size, act='relu', name='out_fc1')
        self.out_fc2_op = default_fc(2, act='softmax', name='out_fc2')

    def create_inputs(self, mode):
        """create layers.data here"""
        inputs = OrderedDict()
        data_attributes = copy.deepcopy(self.data_attributes)
        data_attributes['pos'] = {'shape': (-1, 1), 'dtype': 'int64', 'lod_level': 1}
        data_attributes['click_id'] = {'shape': (-1, 1), 'dtype': 'int64', 'lod_level': 1}
        data_attributes['prev_hidden'] = {'shape': (-1, self.hidden_size), 'dtype': 'float32', 'lod_level': 1}

        if mode in ['train', 'test']:
            list_names = self.item_slot_names + self.user_slot_names + ['pos', 'click_id']
        elif mode in ['inference']:
            list_names = self.item_slot_names + self.user_slot_names + ['pos']
        elif mode == 'infer_init':
            list_names = self.user_slot_names
        elif mode == 'infer_onestep':
            list_names = self.item_slot_names + ['pos', 'prev_hidden']
        else:
            raise NotImplementedError(mode)

        for name in list_names:
            p = data_attributes[name]
            inputs[name] = layers.data(name=name, shape=p['shape'], dtype=p['dtype'], lod_level=p['lod_level'])
        return inputs

    def user_encode(self, inputs):
        user_embedding = self._build_embeddings(inputs, self.user_slot_names)
        user_feature = self.user_feature_fc_op(user_embedding)
        return user_feature

    def item_decode(self, inputs, prev_hidden):
        item_embedding = self._build_embeddings(inputs, self.item_slot_names)
        item_fc = self.item_fc_op(item_embedding)
        item_gru = self.item_gru_op(item_fc, h_0=prev_hidden)
        return item_gru

    ### main functions ###

    def forward(self, inputs):
        """forward"""
        user_feature = self.user_encode(inputs)
        item_gru = self.item_decode(inputs, user_feature)
        click_prob = self.out_fc2_op(self.out_fc1_op(item_gru))
        return click_prob

    def infer_init(self, inputs):
        """inference only the init part"""
        user_feature = self.user_encode(inputs)
        return user_feature

    def infer_onestep(self, inputs):
        """inference the gru-unit by one step"""
        prev_hidden = inputs['prev_hidden']
        item_gru = self.item_decode(inputs, prev_hidden)
        click_prob = self.out_fc2_op(self.out_fc1_op(item_gru))
        return item_gru, click_prob

