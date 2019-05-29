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

from paddle import fluid
import parl.layers as layers
from parl.framework.algorithm import Algorithm


class GenAlgorithm(Algorithm):
    """
    For generation tasks
    """
    def __init__(self, model, optimizer, lr=None, hyperparas=None, gpu_id=-1):
        hyperparas = {} if hyperparas is None else hyperparas
        super(GenAlgorithm, self).__init__(model, hyperparas=hyperparas, gpu_id=gpu_id)
        # self.target_model = deepcopy(model)

        self.optimizer = optimizer
        self.lr = lr
        self.gpu_id = gpu_id

    def train(self):
        """train"""
        inputs = self.model.create_inputs(mode='train')
        click_prob = self.model.forward(inputs)
        click_id = inputs['click_id']
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
        click_prob = self.model.forward(inputs)

        fetch_dict = OrderedDict()
        fetch_dict['click_prob'] = click_prob
        fetch_dict['click_id'] = inputs['click_id'] + layers.reduce_mean(click_prob) * 0     # IMPORTANT!!! equals to label = label, otherwise parallel executor won't get this variable
        return {'fetch_dict': fetch_dict}

    def inference(self):
        """inference"""
        inputs = self.model.create_inputs(mode='inference')
        click_prob = self.model.forward(inputs)
        fetch_dict = OrderedDict()
        fetch_dict['click_prob'] = click_prob
        return {'fetch_dict': fetch_dict}

    def infer_init(self):
        """inference only the init part"""
        inputs = self.model.create_inputs(mode='infer_init')
        init_hidden = self.model.infer_init(inputs)
        fetch_dict = OrderedDict()
        fetch_dict['init_hidden'] = init_hidden
        return {'fetch_dict': fetch_dict}

    def infer_onestep(self):
        """inference the gru-unit by one step"""
        inputs = self.model.create_inputs(mode='infer_onestep')
        next_hidden, click_prob = self.model.infer_onestep(inputs)
        fetch_dict = OrderedDict()
        fetch_dict['next_hidden'] = next_hidden
        fetch_dict['click_prob'] = click_prob
        return {'fetch_dict': fetch_dict}


