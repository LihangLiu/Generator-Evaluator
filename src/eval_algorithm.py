"""
Defines the optimizer and the network outputs
"""
#!/usr/bin/env python
# coding=utf8

import os
from os.path import exists
import sys
import numpy as np
from copy import deepcopy
from collections import OrderedDict
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

PARL_DIR = os.environ['PARL_DIR']
sys.path.append(PARL_DIR)

from paddle import fluid
import parl.layers as layers
from parl.framework.algorithm import Algorithm


class EvalAlgorithm(Algorithm):
    """
    For evaluation tasks
    """
    def __init__(self, model, optimizer, lr=None, hyperparas=None, gpu_id=-1):
        hyperparas = {} if hyperparas is None else hyperparas
        super(EvalAlgorithm, self).__init__(model, hyperparas=hyperparas, gpu_id=gpu_id)
        # self.target_model = deepcopy(model)

        self.optimizer = optimizer
        self.lr = lr
        self.gpu_id = gpu_id

    def train(self):
        """train"""
        inputs = self.model.create_inputs(mode='train')
        click_id = inputs['click_id']
        click_prob = self.model.forward(inputs, mode='train')
        loss = layers.reduce_mean(layers.cross_entropy(input=click_prob, label=click_id))

        if self.optimizer == 'Adam':
            optimizer = fluid.optimizer.Adam(learning_rate=self.lr, epsilon=1e-4)
        elif self.optimizer == 'SGD':
            optimizer = fluid.optimizer.SGD(learning_rate=self.lr)
        optimizer.minimize(loss)

        fetch_dict = OrderedDict()
        fetch_dict['loss'] = loss             # don't rename 'loss', which will be used in parallel exe in computational task
        fetch_dict['click_prob'] = click_prob
        fetch_dict['click_id'] = click_id
        return {'fetch_dict': fetch_dict}

    def test(self):
        """test"""
        inputs = self.model.create_inputs(mode='test')
        click_id = inputs['click_id'] 
        click_prob = self.model.forward(inputs, mode='test')
        click_id = click_id + layers.reduce_mean(click_prob) * 0     # IMPORTANT!!! equals to label = label, otherwise parallel executor won't get this variable

        fetch_dict = OrderedDict()
        fetch_dict['click_prob'] = click_prob
        fetch_dict['click_id'] = click_id
        return {'fetch_dict': fetch_dict}

    def inference(self):
        """inference"""
        inputs = self.model.create_inputs(mode='inference')
        click_prob = self.model.forward(inputs, mode='test')

        fetch_dict = OrderedDict()
        fetch_dict['click_prob'] = click_prob
        return {'fetch_dict': fetch_dict}





