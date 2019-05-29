"""
configuration file
"""

import os
from os.path import join, dirname, basename
from collections import OrderedDict

DATA_DIR = os.environ.get('DATA_DIR', '')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '')        

class Config(object):
    """
    config file
    """
    def __init__(self, exp):
        self.name = basename(__file__)
        self.exp = exp

        ##########
        # logging
        ##########
        self.summary_dir = join(OUTPUT_DIR, 'logs', exp)
        self.model_dir = join(OUTPUT_DIR, 'persistables', exp)

        ####################
        # train settings
        ####################
        self.max_train_steps = 20
        self.prt_interval = 500
        self.optimizer = 'Adam'
        self.lr = 1e-3

        ##########
        # dataset
        ##########
        self.batch_size = 768
        self.train_npz_list = join(DATA_DIR, 'train_npz_list.txt')
        self.test_npz_list = join(DATA_DIR, 'test_npz_list.txt')
        self.npz_config_path = join(DATA_DIR, 'conf/npz_config.json')

        ##########
        # fluid model
        ##########
        ### definitions
        item_attributes = self._get_item_attributes()
        user_attributes = self._get_user_attributes()
        label_attributes = self._get_label_attributes()

        ### used to build layers.data
        self.item_slot_names = item_attributes.keys()
        self.user_slot_names = user_attributes.keys()

        ### used as reference to build fluid.layers.data
        self.data_attributes = OrderedDict()
        self.data_attributes.update(item_attributes)
        self.data_attributes.update(user_attributes)
        self.data_attributes.update(label_attributes)

        ### passed to NpzDataset
        self.requested_npz_names = self.data_attributes.keys()

    def _get_item_attributes(self):
        od = OrderedDict()
        od['category_id'] = {'shape': (-1, 1,), 'dtype': 'int64', 'lod_level': 1}
        od['item_type_id'] = {'shape': (-1, 1,), 'dtype': 'int64', 'lod_level': 1}
        od['layout_id'] = {'shape': (-1, 1,), 'dtype': 'int64', 'lod_level': 1}
        od['mark_id'] = {'shape': (-1, 1,), 'dtype': 'int64', 'lod_level': 1}
        return od

    def _get_user_attributes(self):
        od = OrderedDict()
        od['cuid'] = {'shape': (-1, 1,), 'dtype': 'int64', 'lod_level': 1}
        od['province_id'] = {'shape': (-1, 1,), 'dtype': 'int64', 'lod_level': 1}
        od['net_type_id'] = {'shape': (-1, 1,), 'dtype': 'int64', 'lod_level': 1}
        return od

    def _get_label_attributes(self):
        od = OrderedDict()
        od['click_id'] = {'shape': (-1, 1,), 'dtype': 'int64', 'lod_level': 1}
        return od

