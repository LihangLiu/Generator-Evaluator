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

from fluid_utils import fluid_sequence_pad, fluid_sequence_get_pos, fluid_sequence_index, fluid_sequence_scatter
from base_net import BaseModel, default_fc, default_batch_norm, default_embedding, default_drnn


class RLUniRNN(BaseModel):
    """
    """
    def __init__(self, conf, npz_config, candidate_encode=None):
        super(RLUniRNN, self).__init__(conf, npz_config)
        self._candidate_encode = candidate_encode

        self._create_params()

    def _create_params(self):
        ### embed
        self.dict_data_embed_op = {}
        list_names = self.item_slot_names + self.user_slot_names + ['pos']
        for name in list_names:
            vob_size = self.npz_config['embedding_size'][name] + 1
            self.dict_data_embed_op[name] = default_embedding([vob_size, self.embed_size], 'embed_' + name)

        self.user_feature_fc_op = default_fc(self.hidden_size, act='relu', name='user_feature_fc')

        self.item_fc_op = default_fc(self.hidden_size, act='relu', name='item_fc')
        self.item_gru_fc_op = default_fc(self.hidden_size * 3, act='relu', name='item_gru_fc')
        self.item_gru_op = default_drnn(self.hidden_size, name='item_gru')

        self.out_Q_fc1_op = default_fc(self.hidden_size, act='relu', name='out_Q_fc1')
        self.out_Q_fc2_op = default_fc(1, act=None, name='out_Q_fc2')

    def create_inputs(self, mode):
        """create layers.data here"""
        inputs = OrderedDict()
        data_attributes = copy.deepcopy(self.data_attributes)
        data_attributes['eps'] = {'shape': (1,), 'dtype': 'float32', 'lod_level': 0}
        data_attributes['decode_len'] = {'shape': (-1, 1), 'dtype': 'int64', 'lod_level': 1}
        data_attributes['reward'] = {'shape': (-1, 1), 'dtype': 'float32', 'lod_level': 1}

        if mode in ['train', 'test']:
            list_names = self.item_slot_names + self.user_slot_names + ['decode_len', 'reward']
        elif mode in ['inference']:
            list_names = self.item_slot_names + self.user_slot_names + ['decode_len']
        elif mode == 'sampling':
            list_names = self.item_slot_names + self.user_slot_names + ['decode_len', 'eps']
        else:
            raise NotImplementedError(mode)
            
        for name in list_names:
            p = data_attributes[name]
            inputs[name] = layers.data(name=name, shape=p['shape'], dtype=p['dtype'], lod_level=p['lod_level'])
        return inputs

    def train_rnn(self, item_fc, h_0, pos, pos_embed, output_type=''):
        drnn = fluid.layers.DynamicRNN()
        with drnn.block():
            cur_item_fc = drnn.step_input(item_fc)
            cur_pos_embed = drnn.step_input(pos_embed)
            cur_h_0 = drnn.memory(init=h_0, need_reorder=True)

            # step_input will remove lod info
            cur_item_fc = layers.lod_reset(cur_item_fc, cur_h_0)
            cur_pos_embed = layers.lod_reset(cur_pos_embed, cur_h_0)

            next_h_0, Q = self.sampling_rnn_forward(cur_item_fc, cur_h_0, cur_pos_embed)
            # next_h_0.stop_gradient = True
            # Q.stop_gradient = True
            # layers.Print(next_h_0, summarize=32, message='next_h_0')
            # layers.Print(Q, summarize=32, message='Q')

            # gru_input = self.item_gru_fc_op(layers.concat([cur_item_fc, cur_pos_embed], 1))
            # next_h_0 = self.item_gru_op(gru_input, h_0=cur_h_0)

            if output_type == 'c_Q':
                # Q = self.out_Q_fc2_op(self.out_Q_fc1_op(next_h_0))
                # layers.Print(next_h_0, summarize=32, message='next_h_0 2')
                # layers.Print(Q, summarize=32, message='Q 2')
                drnn.output(Q)

            elif output_type == 'max_Q':
                # e.g. batch_size = 2
                # cur_h_0: lod = [0,1,2]
                cur_pos = drnn.step_input(pos)
                pos = drnn.static_input(pos)            # lod = [0,4,7]
                item_fc = drnn.static_input(item_fc)    # lod = [0,4,7]

                # expand
                expand_h_0 = layers.sequence_expand(cur_h_0, item_fc)               # lod = [0,1,2,3,4,5,6,7]
                expand_pos_embed = layers.sequence_expand(cur_pos_embed, item_fc)   # lod = [0,1,2,3,4,5,6,7]
                expand_item_fc = layers.lod_reset(item_fc, expand_h_0)
                # forward
                _, expand_scores = self.sampling_rnn_forward(expand_item_fc, expand_h_0, expand_pos_embed)
                # reset result lod
                expand_Q = layers.lod_reset(expand_scores, item_fc)            # lod = [0,4,7]             

                cur_step_id = layers.slice(cur_pos, axes=[0, 1], starts=[0, 0], ends=[1, 1])
                mask = layers.cast(pos >= cur_step_id, 'float32')
                expand_Q = expand_Q * mask
                max_Q = layers.sequence_pool(expand_Q, 'max')                       # lod = [0,1,2]
                drnn.output(max_Q)

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

    def sampling_rnn_forward(self, independent_item_fc, independent_hidden, independent_pos_embed):
        gru_input = self.item_gru_fc_op(layers.concat([independent_item_fc, independent_pos_embed], 1))
        item_gru = self.item_gru_op(gru_input, h_0=independent_hidden)
        Q = self.out_Q_fc2_op(self.out_Q_fc1_op(item_gru))
        scores = Q
        return item_gru, scores

    ### main functions ###

    def forward(self, inputs, output_type):
        """forward"""
        decode_len = inputs['decode_len']
        user_feature = self.user_encode(inputs)
        item_embedding = self._build_embeddings(inputs, self.item_slot_names)            
        item_fc = self.item_fc_op(item_embedding)
        pos = fluid_sequence_get_pos(item_fc)
        pos_embed = self.dict_data_embed_op['pos'](pos)

        item_Q = self.train_rnn(item_fc, user_feature, pos, pos_embed, output_type=output_type)
        item_Q = self._cut_by_decode_len(layers.lod_reset(item_Q, item_fc), decode_len)
        return item_Q

    def sampling(self, inputs):
        decode_len = inputs['decode_len']
        user_feature = self.user_encode(inputs)
        item_embedding = self._build_embeddings(inputs, self.item_slot_names)            
        item_fc = self.item_fc_op(item_embedding)
        pos = fluid_sequence_get_pos(item_fc)
        pos_embed = self.dict_data_embed_op['pos'](pos)

        sampled_id = self.sampling_rnn(item_fc, 
                                        h_0=user_feature, 
                                        pos_embed=pos_embed, 
                                        forward_func=self.sampling_rnn_forward, 
                                        sampling_type='eps_greedy',
                                        eps=inputs['eps'])
        sampled_id = self._cut_by_decode_len(layers.lod_reset(sampled_id, item_fc), decode_len)
        return sampled_id



