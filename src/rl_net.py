"""
Defines the network
"""
# coding: utf-8
import os
from os.path import exists
import sys
import json
import copy
import numpy as np
from collections import OrderedDict

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr

PARL_DIR = os.environ['PARL_DIR']
sys.path.append(PARL_DIR)

import parl.layers as layers
from parl.framework.algorithm import Model

from fluid_utils import fluid_sequence_pad, fluid_sequence_get_pos
from base_net import BaseModel, default_fc, default_batch_norm, default_embedding, default_drnn


class RLUniRNN(BaseModel):
    """
    """
    def __init__(self, conf, npz_config):
        super(RLUniRNN, self).__init__(conf, npz_config)
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
        self.item_gru_op = default_drnn(self.hidden_size, name='item_gru')

        self.out_Q_fc1_op = default_fc(self.hidden_size, act='relu', name='out_Q_fc1')
        self.out_Q_fc2_op = default_fc(1, act=None, name='out_Q_fc2')

    def create_inputs(self, mode):
        """create layers.data here"""
        inputs = OrderedDict()
        data_attributes = copy.deepcopy(self.data_attributes)
        data_attributes['pos'] = {'shape': (-1, 1), 'dtype': 'int64', 'lod_level': 1}
        data_attributes['decode_len'] = {'shape': (-1, 1), 'dtype': 'int64', 'lod_level': 1}
        data_attributes['reward'] = {'shape': (-1, 1), 'dtype': 'float32', 'lod_level': 1}
        data_attributes['prev_hidden'] = {'shape': (-1, self.hidden_size), 'dtype': 'float32', 'lod_level': 1}

        if mode in ['train', 'test']:
            list_names = self.item_slot_names + self.user_slot_names + ['pos', 'decode_len', 'reward']
        elif mode in ['inference']:
            list_names = self.item_slot_names + self.user_slot_names + ['pos', 'decode_len']
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

    def custom_rnn(self, item_fc, h_0, decode_len=None, output_type=''):
        drnn = fluid.layers.DynamicRNN()
        pos = fluid_sequence_get_pos(item_fc)
        if decode_len is None:
            decode_ref_fc = item_fc
        else:
            decode_ref_fc = layers.sequence_unpad(fluid_sequence_pad(item_fc, 0), decode_len)
        with drnn.block():
            _ = drnn.step_input(decode_ref_fc)      # to decide decode steps only
            cur_item_fc = drnn.step_input(item_fc)
            cur_h_0 = drnn.memory(init=h_0, need_reorder=True)
            item_fc = drnn.static_input(item_fc)
            pos = drnn.static_input(pos)

            cur_item_fc = layers.lod_reset(cur_item_fc, cur_h_0)
            next_h_0 = self.item_gru_op(cur_item_fc, h_0=cur_h_0)

            if output_type == 'c_Q':
                Q = self.out_Q_fc2_op(self.out_Q_fc1_op(next_h_0))
                drnn.output(Q)

            elif output_type == 'max_Q':
                # e.g. batch_size = 2
                # item_fc: lod = [0,4,7]
                # cur_h_0: lod = [0,1,2]
                cur_step = drnn.memory(shape=[1], dtype='int64', value=0)

                expand_h_0 = layers.sequence_expand(cur_h_0, item_fc)               # lod = [0,1,2,3,4,5,6,7]
                new_item_fc = layers.lod_reset(item_fc, expand_h_0)                 # lod = [0,1,2,3,4,5,6,7]
                next_expand_h_0 = self.item_gru_op(new_item_fc, expand_h_0)         # lod = [0,1,2,3,4,5,6,7]
                next_expand_h_0 = layers.lod_reset(next_expand_h_0, item_fc)        # lod = [0,4,7]

                expand_Q = self.out_Q_fc2_op(self.out_Q_fc1_op(next_expand_h_0))
                cur_step_id = layers.slice(cur_step, axes=[0, 1], starts=[0, 0], ends=[1, 1])
                mask = layers.cast(pos >= cur_step_id, 'float32')
                expand_Q = expand_Q * mask
                max_Q = layers.sequence_pool(expand_Q, 'max')                       # lod = [0,1,2]
                drnn.output(max_Q)

                # update
                next_step = cur_step + 1
                drnn.update_memory(cur_step, next_step)

            elif output_type == 'hidden':
                drnn.output(next_h_0)                

            else:
                raise NotImplementedError(output_type)

            # update
            drnn.update_memory(cur_h_0, next_h_0)

        drnn_output = drnn()
        return drnn_output

    def user_encode(self, inputs):
        user_embedding = self._build_embeddings(inputs, self.user_slot_names)
        user_feature = self.user_feature_fc_op(user_embedding)
        return user_feature

    def item_decode(self, inputs, prev_hidden, output_type):
        item_embedding = self._build_embeddings(inputs, self.item_slot_names + ['pos'])
        item_fc = self.item_fc_op(item_embedding)
        if 'decode_len' in inputs:
            item_gru = self.custom_rnn(item_fc, h_0=prev_hidden, output_type=output_type, decode_len=inputs['decode_len'])
        else:
            item_gru = self.custom_rnn(item_fc, h_0=prev_hidden, output_type=output_type)
        return item_gru

    ### main functions ###

    def forward(self, inputs, output_type):
        """forward"""
        assert output_type in ['c_Q', 'max_Q'], (output_type)
        user_feature = self.user_encode(inputs)
        item_Q = self.item_decode(inputs, user_feature, output_type)
        return item_Q

    def infer_init(self, inputs):
        """inference only the init part"""
        user_feature = self.user_encode(inputs)
        return user_feature

    def infer_onestep(self, inputs):
        """inference the gru-unit by one step"""
        prev_hidden = inputs['prev_hidden']
        item_hidden = self.item_decode(inputs, prev_hidden, output_type='hidden')
        item_Q = self.out_Q_fc2_op(self.out_Q_fc1_op(item_hidden))
        return item_hidden, item_Q


