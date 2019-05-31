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

        self.item_fc_op = default_fc(self.hidden_size, act='relu', name='item_fc')
        self.item_gru_fc_op = default_fc(self.hidden_size * 3, act='relu', name='item_gru_fc')
        self.item_gru_op = default_drnn(self.hidden_size, name='item_gru')

        self.out_Q_fc1_op = default_fc(self.hidden_size, act='relu', name='out_Q_fc1')
        self.out_Q_fc2_op = default_fc(1, act=None, name='out_Q_fc2')

    def create_inputs(self, mode):
        """create layers.data here"""
        inputs = OrderedDict()
        data_attributes = copy.deepcopy(self.data_attributes)
        data_attributes['decode_len'] = {'shape': (-1, 1), 'dtype': 'int64', 'lod_level': 1}
        data_attributes['reward'] = {'shape': (-1, 1), 'dtype': 'float32', 'lod_level': 1}

        if mode in ['train', 'test']:
            list_names = self.item_slot_names + self.user_slot_names + ['decode_len', 'reward']
        elif mode in ['inference', 'sampling']:
            list_names = self.item_slot_names + self.user_slot_names + ['decode_len']
        else:
            raise NotImplementedError(mode)
            
        for name in list_names:
            p = data_attributes[name]
            inputs[name] = layers.data(name=name, shape=p['shape'], dtype=p['dtype'], lod_level=p['lod_level'])
        return inputs

    def train_rnn(self, item_fc, h_0, pos, pos_embed, output_type=''):
        # pos = fluid_sequence_get_pos(item_fc)                       # (b*seq_len, 1)
        # pos_embed = self.dict_data_embed_op['pos'](pos)             # (b*seq_len, dim)

        drnn = fluid.layers.DynamicRNN()
        with drnn.block():
            cur_item_fc = drnn.step_input(item_fc)
            cur_pos_embed = drnn.step_input(pos_embed)
            cur_h_0 = drnn.memory(init=h_0, need_reorder=True)

            # step_input will remove lod info
            cur_item_fc = layers.lod_reset(cur_item_fc, cur_h_0)
            cur_pos_embed = layers.lod_reset(cur_pos_embed, cur_h_0)

            gru_input = self.item_gru_fc_op(layers.concat([cur_item_fc, cur_pos_embed], 1))
            next_h_0 = self.item_gru_op(gru_input, h_0=cur_h_0)

            if output_type == 'c_Q':
                Q = self.out_Q_fc2_op(self.out_Q_fc1_op(next_h_0))
                drnn.output(Q)

            elif output_type == 'max_Q':
                # e.g. batch_size = 2
                # cur_h_0: lod = [0,1,2]
                cur_pos = drnn.step_input(pos)
                pos = drnn.static_input(pos)            # lod = [0,4,7]
                item_fc = drnn.static_input(item_fc)    # lod = [0,4,7]

                expand_h_0 = layers.sequence_expand(cur_h_0, item_fc)               # lod = [0,1,2,3,4,5,6,7]
                expand_pos_embed = layers.sequence_expand(cur_pos_embed, item_fc)   # lod = [0,1,2,3,4,5,6,7]
                expand_gru_input = self.item_gru_fc_op(layers.concat([item_fc, expand_pos_embed], 1))    # lod = [0,4,7]
                expand_gru_input = layers.lod_reset(expand_gru_input, expand_h_0) # lod = [0,1,2,3,4,5,6,7]
                next_expand_h_0 = self.item_gru_op(expand_gru_input, expand_h_0)    # lod = [0,1,2,3,4,5,6,7]
                next_expand_h_0 = layers.lod_reset(next_expand_h_0, item_fc)        # lod = [0,4,7]

                expand_Q = self.out_Q_fc2_op(self.out_Q_fc1_op(next_expand_h_0))
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

    # def _eps_greedy_sampling(self, scores, mask, eps):
    #     scores = scores * mask
    #     scores_padded = layers.squeeze(fluid_sequence_pad(scores, 0, maxlen=128), [2])  # (b*s, 1) -> (b, s, 1) -> (b, s)
    #     mask_padded = layers.squeeze(fluid_sequence_pad(mask, 0, maxlen=128), [2])

    #     def get_greedy_prob(scores_padded, mask_padded):
    #         s = scores_padded - (mask_padded*(-1) + 1) * self.BIG_VALUE
    #         max_value = layers.reduce_max(s, dim=1, keep_dim=True)
    #         greedy_prob = layers.cast(s >= max_value, 'float32')
    #         return greedy_prob
    #     greedy_prob = get_greedy_prob(scores_padded, mask_padded)
    #     eps_prob = mask_padded * eps / layers.reduce_sum(mask_padded, dim=1, keep_dim=True)

    #     final_prob = (greedy_prob + eps_prob) * mask_padded
    #     final_prob = final_prob / layers.reduce_sum(final_prob, dim=1, keep_dim=True)

    #     sampled_id = layers.sampling_id(final_prob)
    #     return layers.cast(layers.reshape(sampled_id, [-1, 1]), 'int64')

    # def sampling_rnn(self, item_fc, h_0, sampling_type='eps_greedy'):
    #     pos = fluid_sequence_get_pos(item_fc)                       # (b*seq_len, 1)
    #     pos_embed = self.dict_data_embed_op['pos'](pos)             # (b*seq_len, dim)
    #     mask = layers.reduce_sum(item_fc, dim=1, keep_dim=True) * 0 + 1

    #     drnn = fluid.layers.DynamicRNN()
    #     with drnn.block():
    #         # e.g. batch_size = 2
    #         _ = drnn.step_input(item_fc)
    #         cur_pos_embed = drnn.step_input(pos_embed)          # lod = []
    #         cur_h_0 = drnn.memory(init=h_0, need_reorder=True)  # lod = [0,1,2]
    #         item_fc = drnn.static_input(item_fc)
    #         mask = drnn.memory(init=mask, need_reorder=True)

    #         # step_input will remove lod info
    #         cur_pos_embed = layers.lod_reset(cur_pos_embed, cur_h_0)

    #         expand_h_0 = layers.sequence_expand(cur_h_0, item_fc)               # lod = [0,1,2,3,4,5,6,7]
    #         expand_pos_embed = layers.sequence_expand(cur_pos_embed, item_fc)   # lod = [0,1,2,3,4,5,6,7]
    #         expand_gru_input = self.item_gru_fc_op(layers.concat([item_fc, expand_pos_embed], 1))    # lod = [0,4,7]
    #         expand_gru_input = layers.lod_reset(expand_gru_input, expand_h_0) # lod = [0,1,2,3,4,5,6,7]
    #         next_expand_h_0 = self.item_gru_op(expand_gru_input, expand_h_0)    # lod = [0,1,2,3,4,5,6,7]
    #         next_expand_h_0 = layers.lod_reset(next_expand_h_0, item_fc)        # lod = [0,4,7]

    #         expand_Q = self.out_Q_fc2_op(self.out_Q_fc1_op(next_expand_h_0))

    #         if sampling_type == 'eps_greedy':
    #             selected_index = self._eps_greedy_sampling(expand_Q, mask, eps=0)

    #         drnn.output(selected_index)

    #         next_h_0 = fluid_sequence_index(next_expand_h_0, selected_index)
    #         next_mask = fluid_sequence_scatter(mask, layers.reshape(selected_index, [-1]), 0.0)

    #         # update
    #         drnn.update_memory(cur_h_0, next_h_0)
    #         drnn.update_memory(mask, next_mask)

    #     drnn_output = drnn()
    #     return drnn_output

    def user_encode(self, inputs):
        user_embedding = self._build_embeddings(inputs, self.user_slot_names)
        user_feature = self.user_feature_fc_op(user_embedding)
        return user_feature

    # def item_decode(self, inputs, prev_hidden, output_type):
    #     decode_len = inputs['decode_len']
    #     item_embedding = self._build_embeddings(inputs, self.item_slot_names)
    #     item_fc = self.item_fc_op(item_embedding)
    #     if output_type in ['c_Q', 'max_Q']:
    #         item_gru = self.train_rnn(item_fc, h_0=prev_hidden, output_type=output_type)
    #     elif output_type == 'sampled_id':
    #         item_gru = self.sampling_rnn(item_fc, h_0=prev_hidden)
    #     item_gru = self._cut_by_decode_len(layers.lod_reset(item_gru, item_fc), decode_len)
    #     return item_gru

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
                                        sampling_type='eps_greedy')
        sampled_id = self._cut_by_decode_len(layers.lod_reset(sampled_id, item_fc), decode_len)
        return sampled_id



