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

from fluid_utils import fluid_sequence_pad, fluid_split, fluid_sequence_get_seq_len, fluid_sequence_get_pos
from base_net import BaseModel, default_fc, default_batch_norm, default_embedding, default_drnn

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # filename='hot_rl.log', 


class BiRNN(BaseModel):
    """
    """
    def __init__(self, conf, npz_config):
        super(BiRNN, self).__init__(conf, npz_config)
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
        self.item_gru_forward_op = default_drnn(self.hidden_size, name='item_gru_forward')
        self.item_gru_backward_op = default_drnn(self.hidden_size, name='item_gru_backward', is_reverse=True)

        self.out_click_fc1_op = default_fc(self.hidden_size, act='relu', name='out_click_fc1')
        self.out_click_fc2_op = default_fc(2, act='softmax', name='out_click_fc2')

    def create_inputs(self, mode):
        """create layers.data here"""
        inputs = OrderedDict()
        data_attributes = copy.deepcopy(self.data_attributes)
        data_attributes['click_id'] = {'shape': (-1, 1), 'dtype': 'int64', 'lod_level': 1}

        if mode in ['train', 'test']:
            list_names = self.item_slot_names + self.user_slot_names + ['click_id']
        elif mode in ['inference']:
            list_names = self.item_slot_names + self.user_slot_names
        else:
            raise NotImplementedError(mode)

        for name in list_names:
            p = data_attributes[name]
            inputs[name] = layers.data(name=name, shape=p['shape'], dtype=p['dtype'], lod_level=p['lod_level'])
        return inputs

    def forward(self, inputs, mode):
        """forward"""
        # encode
        user_embedding = self._build_embeddings(inputs, self.user_slot_names)
        user_feature = self.user_feature_fc_op(user_embedding)
        # item embed + pos embed
        item_embedding = self._build_embeddings(inputs, self.item_slot_names)            
        item_fc = self.item_fc_op(item_embedding)
        pos = fluid_sequence_get_pos(item_fc)
        pos_embed = self.dict_data_embed_op['pos'](pos)

        # item gru
        gru_input = self.item_gru_fc_op(layers.concat([item_fc, pos_embed], 1))
        item_gru_forward = self.item_gru_forward_op(gru_input, h_0=user_feature)
        item_gru_backward = self.item_gru_backward_op(gru_input, h_0=user_feature)
        item_gru = layers.concat([item_gru_forward, item_gru_backward], axis=1)
        click_prob = self.out_click_fc2_op(self.out_click_fc1_op(item_gru))
        return click_prob


class MultiHeadAttention(Model):
    def __init__(self, nf, name=''):
        super(MultiHeadAttention, self).__init__()
        self._nf = nf
        self._safe_eps = 1e-5

        self.Q_fc_op = default_fc(nf, num_flatten_dims=2, act='relu', name='%s_Q_fc' % name)
        self.K_fc_op = default_fc(nf, num_flatten_dims=2, act='relu', name='%s_K_fc' % name)
        self.V_fc_op = default_fc(nf, num_flatten_dims=2, act='relu', name='%s_V_fc' % name)

    def get_input_specs(self):
        return []

    def get_action_specs(self):
        return []
        
    def __call__(self, input_Q, input_K, input_V, num_head, mask):
        """
        args:
            input_Q: (batch, max_num0, dim0)
            input_K: (batch, max_num1, dim1)
            input_V: (batch, max_num1, dim2)
            mask: (batch, max_num0, max_num1)
        returns:
            output: (batch, max_num0, dim)
        """
        # fcs
        Q = self.Q_fc_op(input_Q)
        K = self.K_fc_op(input_K)
        V = self.V_fc_op(input_V)

        # multi head
        dim = Q.shape[-1]
        assert dim % num_head == 0, (dim, num_head)
        list_output = []
        sub_Qs = fluid_split(Q, num_head, 2)
        sub_Ks = fluid_split(K, num_head, 2)
        sub_Vs = fluid_split(V, num_head, 2)
        for head_id in range(num_head):
            sub_Q = sub_Qs[head_id]                         # (batch, max_num, dim/num_head)
            sub_K = sub_Ks[head_id]
            sub_V = sub_Vs[head_id]
            # matmul -> scale -> mask -> softmax -> mask -> /sum
            Q_K_T = layers.matmul(sub_Q, layers.transpose(sub_K, perm=[0, 2, 1]))   # (batch, max_num0, max_num1)
            Q_K_T = Q_K_T / np.sqrt(self._nf)
            Q_K_T = Q_K_T * mask

            Q_K_T = layers.softmax(Q_K_T)
            Q_K_T = Q_K_T * mask

            Q_K_T = Q_K_T / (layers.reduce_sum(Q_K_T, dim=2, keep_dim=True) + self._safe_eps)
            
            # weighted sum
            atten_out = layers.matmul(Q_K_T, sub_V)         # (batch, max_num0, dim/num_head)
            list_output.append(atten_out)
        output = layers.concat(list_output, 2)
        return output


class Transformer(BaseModel):
    """
    max_seq_len=None for dynamically calculating max(seq_lens) for each batch.
    """
    def __init__(self, conf, npz_config, num_blocks=None, num_head=None):
        super(Transformer, self).__init__(conf, npz_config)

        self._num_blocks = num_blocks
        self._num_head = num_head
        self._dropout_prob = 0.1
        self._max_seq_len = 64
        
        self._create_params()

    def _create_params(self):
        """create all params here"""
        ### embed
        self.dict_data_embed_op = {}
        list_names = self.item_slot_names + self.user_slot_names + ['pos']
        for name in list_names:
            vob_size = self.npz_config['embedding_size'][name] + 1
            self.dict_data_embed_op[name] = default_embedding([vob_size, self.embed_size], 'embed_' + name)

        ### embed fc
        self.user_feature_fc_op = default_fc(self.hidden_size, act='relu', name='user_feature_fc')
        self.item_fc_op = default_fc(self.hidden_size, act='relu', name='item_fc')
        self.input_embed_fc_op = default_fc(self.hidden_size, act='relu', name='input_embed_fc')
            
        ### blocks
        self.atten_ops = []
        self.atten_norm_ops = []
        self.ffn_fc1_ops = []
        self.ffn_fc2_ops = []
        self.ffn_norm_ops = []
        for block_id in range(self._num_blocks):
            # attention
            self.atten_ops.append(MultiHeadAttention(self.hidden_size, name="blk%d_atten" % block_id))
            self.atten_norm_ops.append(default_batch_norm(name="blk%d_norm" % block_id))
            # feed forward
            self.ffn_fc1_ops.append(default_fc(self.hidden_size * 2, act='relu', name='blk%d_ffn_fc1' % block_id))
            self.ffn_fc2_ops.append(default_fc(self.hidden_size, act='relu', name='blk%d_ffn_fc2' % block_id))
            self.ffn_norm_ops.append(default_batch_norm(name='blk%d_ffn_norm' % block_id))

        ### output
        self.output_fc1_op = default_fc(self.hidden_size, act='relu', name='output_fc1')
        self.output_fc2_op = default_fc(2, act='softmax', name='output_fc2')

    def _attention_norm(self, is_test, atten_op, norm_op, 
                        input, atten_input, max_seq_len, max_atten_seq_len,
                        num_head):
        """
        If QK_type is relu or softsign, no need to use maxlen for sequence_pad.
        args:
            input: (batch*seq_len, dim), 1-level lod tensor
            atten_input: (batch*atten_seq_len, dim), 1-level lod tensor
        returns:
            output: (batch*seq_len, dim), 1-level lod tensor
        """
        def get_seq_len_mask(input, atten_input, max_seq_len, max_atten_seq_len):
            ones = layers.reduce_sum(input, dim=1, keep_dim=True) * 0 + 1               # (batch*seq_len, 1)
            atten_ones = layers.reduce_sum(atten_input, dim=1, keep_dim=True) * 0 + 1
            ones_padded = fluid_sequence_pad(ones, 0, max_seq_len)                      # (batch, seq_len, 1)
            atten_ones_padded = fluid_sequence_pad(atten_ones, 0, max_atten_seq_len)
            seq_len_mask = layers.matmul(ones_padded, layers.transpose(atten_ones_padded, perm=[0, 2, 1]))
            seq_len_mask.stop_gradient = True
            return seq_len_mask         # (batch, seq_len, atten_seq_len)

        seq_lens = fluid_sequence_get_seq_len(input)
        ### padding
        input_padded = fluid_sequence_pad(input, 0, max_seq_len) # (batch, max_seq_len, dim)
        atten_input_padded = fluid_sequence_pad(atten_input, 0, max_atten_seq_len) # (batch, max_recent_seq_len, dim)
        mask = get_seq_len_mask(input, atten_input, max_seq_len, max_atten_seq_len)
        atten_out = atten_op(input_padded, atten_input_padded, atten_input_padded, num_head, mask)
        ### flatten and unpad
        output = layers.sequence_unpad(atten_out, seq_lens)
        ### residual and normalize
        output = norm_op(input + output, is_test)
        output = layers.dropout(output, dropout_prob=self._dropout_prob, is_test=is_test)
        return output

    def _ffn(self, is_test, fc1_op, fc2_op, norm_op, input):
        ### fcs
        output = fc2_op(fc1_op(input))
        ### normalize
        output = norm_op(output, is_test)
        output = layers.dropout(output, dropout_prob=self._dropout_prob, is_test=is_test)
        return output

    def create_inputs(self, mode):
        """create layers.data here"""
        inputs = OrderedDict()
        data_attributes = copy.deepcopy(self.data_attributes)
        data_attributes['click_id'] = {'shape': (-1, 1), 'dtype': 'int64', 'lod_level': 1}

        if mode in ['train', 'test']:
            list_names = self.item_slot_names + self.user_slot_names + ['click_id']
        elif mode in ['inference']:
            list_names = self.item_slot_names + self.user_slot_names
        else:
            raise NotImplementedError(mode)

        for name in list_names:
            p = data_attributes[name]
            inputs[name] = layers.data(name=name, shape=p['shape'], dtype=p['dtype'], lod_level=p['lod_level'])
        return inputs

    def transformer_decode(self, is_test, trans_in):
        block_in = trans_in
        for b_id in range(self._num_blocks):
            ### attentions
            atten_out = self._attention_norm(is_test, 
                                            self.atten_ops[b_id],
                                            self.atten_norm_ops[b_id],
                                            block_in, 
                                            block_in,
                                            self._max_seq_len,
                                            self._max_seq_len,
                                            self._num_head)
            ### feed forward
            ffn_out = self._ffn(is_test, 
                                self.ffn_fc1_ops[b_id], 
                                self.ffn_fc2_ops[b_id], 
                                self.ffn_norm_ops[b_id],
                                atten_out)
            ffn_out = block_in + ffn_out
            ### for next
            block_in = ffn_out
        return ffn_out

    def forward(self, inputs, mode):
        """
        don't use sequence_expand for backward
        """
        is_test = True if (mode in ['test', 'inference']) else False        

        # encode
        user_embedding = self._build_embeddings(inputs, self.user_slot_names)
        user_feature = self.user_feature_fc_op(user_embedding)

        # item embed and pos embed
        item_embedding = self._build_embeddings(inputs, self.item_slot_names)
        item_fc = self.item_fc_op(item_embedding)
        pos = fluid_sequence_get_pos(item_fc)
        pos_embed = self.dict_data_embed_op['pos'](pos)
        input_embed = layers.concat([item_fc, pos_embed,
                                    layers.sequence_expand_as(user_feature, item_fc)], 
                                    1)

        # transformer
        trans_in = self.input_embed_fc_op(input_embed)
        decoding = self.transformer_decode(is_test, trans_in)
        click_prob = self.output_fc2_op(self.output_fc1_op(decoding))
        return click_prob







