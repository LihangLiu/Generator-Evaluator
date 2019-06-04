"""
Defines the optimizer and the network outputs
"""
#!/usr/bin/env python
# coding=utf8

import os
from os.path import exists
import sys
from collections import OrderedDict
import numpy as np
from copy import deepcopy
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

PARL_DIR = os.environ['PARL_DIR']
sys.path.append(PARL_DIR)

from paddle.fluid.executor import _fetch_var
from paddle import fluid
import parl.layers as layers
from parl.framework.algorithm import Algorithm

from utils import get_boundaries_and_values_for_piecewise_decay
from fluid_utils import fluid_sequence_delay


class RLAlgorithm(Algorithm):
    """
    For generation tasks
    """
    def __init__(self, model, optimizer, lr=None, hyperparas=None, gpu_id=-1,
                    gamma=None, target_update_ratio=0.01):
        hyperparas = {} if hyperparas is None else hyperparas
        super(RLAlgorithm, self).__init__(model, hyperparas=hyperparas, gpu_id=gpu_id)
        self.target_model = deepcopy(model)

        self.optimizer = optimizer
        self.lr = lr
        self._gamma = gamma
        self.target_update_ratio = target_update_ratio
        self._learn_cnt = 0

        self.gpu_id = gpu_id

    def get_target_Q(self, max_Q, reward):
        """reward: used to recover lod_level"""
        next_max_Q = fluid_sequence_delay(reward * 0 + max_Q, OOV=0)  # TODO, use "reward * 0" to recover lod_level in infer stage
        target_Q = reward + self._gamma * next_max_Q
        target_Q.stop_gradient = True
        return target_Q

    def train(self):
        """train"""
        inputs = self.model.create_inputs(mode='train')
        reward = layers.cast(inputs['reward'], 'float32')

        c_Q = self.model.forward(inputs, output_type='c_Q')
        max_Q = self.target_model.forward(inputs, output_type='max_Q')
        target_Q = self.get_target_Q(max_Q, reward)
        loss = layers.reduce_mean(layers.square_error_cost(c_Q, target_Q))

        if self.optimizer == 'Adam':
            optimizer = fluid.optimizer.Adam(learning_rate=self.lr, epsilon=1e-4)
        elif self.optimizer == 'SGD':
            optimizer = fluid.optimizer.SGD(learning_rate=self.lr)
        optimizer.minimize(loss)

        fetch_dict = OrderedDict()
        fetch_dict['loss'] = loss             # don't rename 'loss', which will be used in parallel exe in computational task
        fetch_dict['c_Q'] = c_Q
        fetch_dict['reward'] = reward
        return {'fetch_dict': fetch_dict}

    def test(self):
        """test"""
        inputs = self.model.create_inputs(mode='train')
        reward = layers.cast(inputs['reward'], 'float32')

        c_Q = self.model.forward(inputs, output_type='c_Q')
        max_Q = self.target_model.forward(inputs, output_type='max_Q')
        target_Q = self.get_target_Q(max_Q, reward)

        loss = layers.reduce_mean(layers.square_error_cost(c_Q, target_Q))

        fetch_dict = OrderedDict()
        fetch_dict['loss'] = loss
        fetch_dict['c_Q'] = c_Q
        fetch_dict['reward'] = reward
        return {'fetch_dict': fetch_dict}

    def inference(self):
        """inference"""
        inputs = self.model.create_inputs(mode='inference')
        c_Q = self.model.forward(inputs, output_type='c_Q')

        fetch_dict = OrderedDict()
        fetch_dict['c_Q'] = c_Q
        return {'fetch_dict': fetch_dict}

    def sampling(self):
        """sampling"""
        inputs = self.model.create_inputs(mode='sampling')
        sampled_id = self.model.sampling(inputs)

        fetch_dict = OrderedDict()
        fetch_dict['sampled_id'] = sampled_id
        return {'fetch_dict': fetch_dict}

    def before_every_batch(self):
        """
        TODO: memory leak caused by np.array(var.get_tensor()) within _fetch_var() 
            (https://github.com/PaddlePaddle/Paddle/issues/17176)
        """
        interval = 100
        if self._learn_cnt % interval == 0:
            logging.info('model sync to target_model %d' % self._learn_cnt)
            self.model.sync_paras_to(self.target_model, self.gpu_id, 1.0)
        self._learn_cnt += 1

        # if self._learn_cnt == 0:
        #     self.model.sync_paras_to(self.target_model, self.gpu_id, 1.0)
        #     self._learn_cnt += 1
        #     return    

        # self.model.sync_paras_to(self.target_model, self.gpu_id, self.target_update_ratio)
        # self._learn_cnt += 1  



