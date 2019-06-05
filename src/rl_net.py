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

from fluid_utils import (fluid_sequence_pad, fluid_sequence_get_pos, fluid_sequence_index, 
                        fluid_sequence_scatter, fluid_sequence_advance)
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
        if self._candidate_encode:
            self.candidate_encode_fc_op = default_fc(self.hidden_size, act='relu', name='candidate_encode_fc')

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

            if output_type == 'c_Q':
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

    def candidate_encode(self, item_fc):
        if self._candidate_encode == 'mean':
            cand_encoding = layers.sequence_pool(item_fc, 'average')
        elif self._candidate_encode == 'sum':
            cand_encoding = layers.sequence_pool(item_fc, 'sum')
        return cand_encoding   

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

        if self._candidate_encode:
            cand_encoding = self.candidate_encode(item_fc)
            init_hidden = self.candidate_encode_fc_op(layers.concat([user_feature, cand_encoding], 1))
        else:
            init_hidden = user_feature
        item_Q = self.train_rnn(item_fc, init_hidden, pos, pos_embed, output_type=output_type)
        item_Q = self._cut_by_decode_len(layers.lod_reset(item_Q, item_fc), decode_len)
        return item_Q

    def sampling(self, inputs):
        decode_len = inputs['decode_len']
        user_feature = self.user_encode(inputs)
        item_embedding = self._build_embeddings(inputs, self.item_slot_names)            
        item_fc = self.item_fc_op(item_embedding)
        pos = fluid_sequence_get_pos(item_fc)
        pos_embed = self.dict_data_embed_op['pos'](pos)

        if self._candidate_encode:
            cand_encoding = self.candidate_encode(item_fc)
            init_hidden = self.candidate_encode_fc_op(layers.concat([user_feature, cand_encoding], 1))
        else:
            init_hidden = user_feature
        sampled_id = self.sampling_rnn(item_fc, 
                                        h_0=init_hidden, 
                                        pos_embed=pos_embed, 
                                        forward_func=self.sampling_rnn_forward, 
                                        sampling_type='eps_greedy',
                                        eps=inputs['eps'])
        sampled_id = self._cut_by_decode_len(layers.lod_reset(sampled_id, item_fc), decode_len)
        return sampled_id



class RLPointerNet(BaseModel):
    """
    """
    def __init__(self, conf, npz_config, candidate_encode, attention_type='dot'):
        super(RLPointerNet, self).__init__(conf, npz_config)
        self._candidate_encode = candidate_encode
        self._attention_type = attention_type
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
        self.atten_item_fc_op = default_fc(self.hidden_size, act='relu', name='atten_item_fc')
        self.candidate_encode_fc_op = default_fc(self.hidden_size, act='relu', name='candidate_encode_fc')

        self.item_gru_fc_op = default_fc(self.hidden_size * 3, act='relu', name='item_gru_fc')
        self.item_gru_op = default_drnn(self.hidden_size, name='item_gru')
        self.hidden_fc_op = default_fc(self.hidden_size, name='hidden_fc')

        if self._attention_type == 'concat_fc':
            self.atten_fc_op = default_fc(1, act=None, name='atten_fc')

        # self.out_Q_fc1_op = default_fc(self.hidden_size, act='relu', name='out_Q_fc1')
        # self.out_Q_fc2_op = default_fc(1, act=None, name='out_Q_fc2')

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

    def _dot_attention(self, input, atten_items):
        """
        args:
            input: (batch, dim), lod_level = 0
            atten_items: (batch*seq_len, dim), lod_level = 1
        return:
            atten_weights: (batch*seq_len, 1), lod_level = 1
        """
        expand_input = layers.sequence_expand(input, atten_items) #(batch*seq_len, dim), lod_level = 0
        expand_input = layers.lod_reset(expand_input, atten_items)    #(batch*seq_len, dim), lod_level = 1
        if self._attention_type == 'concat_fc':
            atten_weights = self.atten_fc_op(layers.concat([expand_input, atten_items], 1))
        elif self._attention_type == 'dot':
            atten_weights = layers.reduce_sum(expand_input * atten_items, dim=1, keep_dim=True)    #(batch*seq_len, 1), lod_level = 1
        return atten_weights

    def train_rnn(self, item_fc, atten_item_fc, h_0, pos, pos_embed, output_type=''):
        shifted_item_fc = fluid_sequence_advance(item_fc, OOV=0)
        drnn = fluid.layers.DynamicRNN()
        with drnn.block():
            cur_item_fc = drnn.step_input(shifted_item_fc)
            cur_pos_embed = drnn.step_input(pos_embed)
            cur_h_0 = drnn.memory(init=h_0, need_reorder=True)

            # step_input will remove lod info
            cur_item_fc = layers.lod_reset(cur_item_fc, cur_h_0)
            cur_pos_embed = layers.lod_reset(cur_pos_embed, cur_h_0)

            next_h_0, hidden_fc = self.sampling_rnn_forward(cur_item_fc, cur_h_0, cur_pos_embed)

            if output_type == 'c_Q':
                cur_atten_item_fc = drnn.step_input(atten_item_fc)
                cur_atten_item_fc = layers.lod_reset(cur_atten_item_fc, cur_h_0)

                Q = layers.reduce_sum(hidden_fc * cur_atten_item_fc, dim=1, keep_dim=True)
                drnn.output(Q)

            elif output_type == 'max_Q':
                cur_pos = drnn.step_input(pos)
                pos = drnn.static_input(pos)
                atten_item_fc = drnn.static_input(atten_item_fc)

                expand_Q = self._dot_attention(hidden_fc, atten_item_fc)


                cur_step_id = layers.slice(cur_pos, axes=[0, 1], starts=[0, 0], ends=[1, 1])
                mask = layers.cast(pos >= cur_step_id, 'float32')
                expand_Q = expand_Q * mask
                max_Q = layers.sequence_pool(expand_Q, 'max')
                drnn.output(max_Q)

            else:
                raise NotImplementedError(output_type)

            # update
            drnn.update_memory(cur_h_0, next_h_0)

        drnn_output = drnn()
        return drnn_output

    def sampling_rnn(self, item_fc, atten_item_fc, h_0, pos_embed, sampling_type, eps=0):
        oov_item_fc = layers.fill_constant_batch_size_like(item_fc, shape=item_fc.shape, value=0, dtype='float32')
        oov_item_fc = layers.lod_reset(oov_item_fc, h_0)
        mask = layers.reduce_sum(item_fc, dim=1, keep_dim=True) * 0 + 1
        drnn = fluid.layers.DynamicRNN()
        with drnn.block():
            _ = drnn.step_input(item_fc)
            cur_pos_embed = drnn.step_input(pos_embed)
            cur_item_fc = drnn.memory(init=oov_item_fc, need_reorder=True)
            cur_h_0 = drnn.memory(init=h_0, need_reorder=True)
            mask = drnn.memory(init=mask, need_reorder=True)
            item_fc = drnn.static_input(item_fc)
            atten_item_fc = drnn.static_input(atten_item_fc)

            # step_input will remove lod info
            cur_pos_embed = layers.lod_reset(cur_pos_embed, cur_h_0)

            next_h_0, hidden_fc = self.sampling_rnn_forward(cur_item_fc, cur_h_0, cur_pos_embed)
            expand_Q = self._dot_attention(hidden_fc, atten_item_fc)

            if sampling_type == 'eps_greedy':
                selected_index = self.eps_greedy_sampling(expand_Q, mask, eps=eps)


            drnn.output(selected_index)

            next_item_fc = fluid_sequence_index(item_fc, selected_index)
            next_mask = fluid_sequence_scatter(mask, layers.reshape(selected_index, [-1]), 0.0)

            # update
            drnn.update_memory(cur_item_fc, next_item_fc)
            drnn.update_memory(cur_h_0, next_h_0)
            drnn.update_memory(mask, next_mask)

        drnn_output = drnn()
        return drnn_output

    def user_encode(self, inputs):
        user_embedding = self._build_embeddings(inputs, self.user_slot_names)
        user_feature = self.user_feature_fc_op(user_embedding)
        return user_feature

    def candidate_encode(self, item_fc):
        if self._candidate_encode == 'mean':
            cand_encoding = layers.sequence_pool(item_fc, 'average')
        elif self._candidate_encode == 'sum':
            cand_encoding = layers.sequence_pool(item_fc, 'sum')
        return cand_encoding   

    def sampling_rnn_forward(self, independent_item_fc, independent_hidden, independent_pos_embed):
        gru_input = self.item_gru_fc_op(layers.concat([independent_item_fc, independent_pos_embed], 1))
        item_gru = self.item_gru_op(gru_input, h_0=independent_hidden)
        hidden_fc = self.hidden_fc_op(item_gru)
        return item_gru, hidden_fc

    ### main functions ###

    def forward(self, inputs, output_type):
        """forward"""
        decode_len = inputs['decode_len']
        user_feature = self.user_encode(inputs)
        item_embedding = self._build_embeddings(inputs, self.item_slot_names)            
        item_fc = self.item_fc_op(item_embedding)
        atten_item_fc = self.atten_item_fc_op(item_embedding)
        cand_encoding = self.candidate_encode(item_fc)
        pos = fluid_sequence_get_pos(item_fc)
        pos_embed = self.dict_data_embed_op['pos'](pos)

        init_hidden = self.candidate_encode_fc_op(layers.concat([user_feature, cand_encoding], 1))
        item_Q = self.train_rnn(item_fc, atten_item_fc, init_hidden, pos, pos_embed, output_type=output_type)
        item_Q = self._cut_by_decode_len(layers.lod_reset(item_Q, item_fc), decode_len)
        return item_Q

    def sampling(self, inputs):
        decode_len = inputs['decode_len']
        user_feature = self.user_encode(inputs)
        item_embedding = self._build_embeddings(inputs, self.item_slot_names)            
        item_fc = self.item_fc_op(item_embedding)
        atten_item_fc = self.atten_item_fc_op(item_embedding)
        cand_encoding = self.candidate_encode(item_fc)
        pos = fluid_sequence_get_pos(item_fc)
        pos_embed = self.dict_data_embed_op['pos'](pos)

        init_hidden = self.candidate_encode_fc_op(layers.concat([user_feature, cand_encoding], 1))
        sampled_id = self.sampling_rnn(item_fc, 
                                        atten_item_fc,
                                        h_0=init_hidden, 
                                        pos_embed=pos_embed, 
                                        sampling_type='eps_greedy',
                                        eps=inputs['eps'])
        sampled_id = self._cut_by_decode_len(layers.lod_reset(sampled_id, item_fc), decode_len)
        return sampled_id
