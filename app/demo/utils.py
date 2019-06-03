import numpy as np
import os
from os.path import join, dirname, exists
import time
import datetime
import collections
import copy
import logging

import tensorflow as tf

import _init_paths

from src.utils import tik, tok, AssertEqual
from data.npz_dataset import FakeTensor

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # filename='hot_rl.log', 


def add_scalar_summary(summary_writer, index, tag, value):
    logging.info("Step {}: {} {}".format(index, tag, value))
    if summary_writer:
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        summary_writer.add_summary(summary, index)
    # else:
    #     if not exists(conf.summary_dir):
    #         os.makedirs(conf.summary_dir)
    #     log_file = join(conf.summary_dir, 'log_%s.txt' % args.exp)
    #     with open(log_file, 'a') as f:
    #         f.write("{} Step {}: {} {}\n".format(datetime.datetime.now(), index, tag, value))


def sequence_unconcat(input_sequence, lens):
    """
    input_sequence: (sum(lens), *)
    e.g.
        input_sequence = [1,2,3,4,5,6]
        lens = [2, 0, 4]
        return [[1,2], [], [3,4,5,6]]
    """
    AssertEqual(len(input_sequence), np.sum(lens))
    res = []
    start = 0
    for l in lens:
        res.append(input_sequence[start: start + l])
        start += l
    return res


def sequence_expand(input, lens):
    """
    input: len() = batch_size, e.g. [(dim), [], (dim), ...]
    lens: (batch_size,)
    e.g.
        input_sequence = [(dim), [], (dim)]
        lens = [1,0,2]
        return (3, dim)
    e.g.
        input_sequence = [(dim), [dim], (dim)]
        lens = [1,0,2]
        return (3, dim)
    """
    AssertEqual(len(input), len(lens))
    res = []
    for inp, l in zip(input, lens):
        if l > 0:
            res.append(np.array([inp] * l))     # (l, dim)
    return np.concatenate(res, axis=0)


def sequence_gather(input_sequence, lens, index):
    """
    input_sequence: (sum(lens), *)
    lens: (batch_size,)
    index: len() = batch_size
    e.g.
        input_sequence = [1,2, 3,4,5,6]
        lens = [2, 0, 4]
        index = [1, None, 2]
        return [2, [], 5]
    """
    AssertEqual(len(input_sequence), np.sum(lens))
    AssertEqual(len(lens), len(index))
    input_unconcat = sequence_unconcat(input_sequence, lens)
    res = []
    for sub_input, sub_index in zip(input_unconcat, index):
        if not sub_index is None:
            res.append(sub_input[sub_index])
        else:
            res.append([])
    return res
    

def sequence_sampling(scores, lens, sampling_type):
    """
    scores: (sum(lens),)
    lens: (batch_size,)
    return: (batch_size,) 
    e.g. 
        scores = [0.4,0.3, 0.4,0.9,0.8]
        lens = [2, 0, 3]
        return [0, None, 1]
    """ 
    AssertEqual(len(scores), np.sum(lens))
    scores_unconcat = sequence_unconcat(scores, lens)
    res_index = []
    for sub_score in scores_unconcat:
        if len(sub_score) > 0:
            if sampling_type == 'greedy':
                selected_index = np.argmax(sub_score)
            res_index.append(selected_index)
        else:
            res_index.append(None)
    return res_index


class BatchData(object):
    def __init__(self, conf, tensor_dict):
        self.conf = conf
        self.tensor_dict = copy.deepcopy(tensor_dict)
        self.align_tensors_with_conf()

        self._decode_len = None

    def seq_lens(self):
        return self.tensor_dict[self.first_item_slot_name()].seq_lens

    def lod(self):
        return self.tensor_dict[self.first_item_slot_name()].lod

    def batch_size(self):
        return len(self.seq_lens())

    def offset(self):
        return self.lod()[0][:-1]

    def pos(self):
        return np.concatenate([np.arange(s) for s in self.seq_lens()], axis=0)

    def total_item_num(self):
        return np.sum(self.seq_lens())

    def decode_len(self):
        return self._decode_len

    def set_decode_len(self, decode_len):
        self._decode_len = np.array(decode_len)

    def first_item_slot_name(self):
        return self.conf.item_slot_names[0]

    def align_tensors_with_conf(self):
        for name, ft in self.tensor_dict.items():
            proper = self.conf.data_attributes[name]
            ft.values = ft.values.astype(proper['dtype'])
            if len(proper['shape']) - len(ft.values.shape) == 1:
                ft.values = np.expand_dims(ft.values, -1)

    def expand_candidates(self, other_batch_data, lens):
        """
        Regard other_batch_data as a candidate pool
        Only expand item-level values
            1. append values of self and other_batch_data
            2. construct index to get new batch_data
        lens: (batch_size,), len to expand
        """
        AssertEqual(len(lens), self.batch_size())

        total_cand_len = other_batch_data.total_item_num()
        total_item_len = self.total_item_num()
        cand_indice = np.arange(total_item_len, total_item_len + total_cand_len)     

        global_item_indice = []
        lod = self.lod()[0]
        for i in range(len(lod) - 1):
            start, end = lod[i], lod[i+1]
            old_indice = np.arange(start, end)
            new_indice = np.random.choice(cand_indice, size=lens[i], replace=False)
            global_item_indice.append(old_indice)
            global_item_indice.append(new_indice)
        global_item_indice = np.concatenate(global_item_indice, axis=0)

        prev_seq_lens = self.seq_lens()
        seq_lens = [s + l for s,l in zip(prev_seq_lens, lens)]
        # update tensor_dict
        for name in self.conf.item_slot_names:
            values = np.concatenate([self.tensor_dict[name].values, other_batch_data.tensor_dict[name].values], 0)
            self.tensor_dict[name] = FakeTensor(values[global_item_indice], seq_lens)

    def get_candidates(self, pre_items, stop_flags=None):
        """
        pre_items: len() = batch_size
        stop_flags: (batch_size,)
        return:
            candidate_items: len() = batch_size, e.g. [[2,3,5], [3,4], ...]
        """
        if stop_flags is None:
            stop_flags = np.zeros([len(pre_items)])
        AssertEqual(len(pre_items), len(stop_flags))

        res = []
        for pre, seq_len, stop in zip(pre_items, self.seq_lens(), stop_flags):
            if stop:
                res.append([])
            else:
                full = np.arange(seq_len)
                res.append(np.setdiff1d(full, pre))
        return res

    def get_reordered(self, order):
        """
        get item-level features by order
        click_id will be removed
        """
        AssertEqual(len(order), self.batch_size())

        global_item_indice = []
        for sub_order, sub_offset in zip(order, self.offset()):
            global_item_indice.append(np.array(sub_order) + sub_offset)
        global_item_indice = np.concatenate(global_item_indice, axis=0)

        new_batch_data = BatchData(self.conf, self.tensor_dict)
        new_seq_lens = [len(od) for od in order]
        for name in new_batch_data.conf.item_slot_names:
            values = new_batch_data.tensor_dict[name].values
            new_batch_data.tensor_dict[name] = FakeTensor(values[global_item_indice], new_seq_lens)
        del new_batch_data.tensor_dict['click_id']
        return new_batch_data

    def get_reordered_keep_candidate(self, order):
        """
        get item-level features by order + rest_items
        click_id will be removed
        """
        rest_items = self.get_candidates(order)
        new_order = [np.append(od, ri) for od, ri in zip(order, rest_items)]
        return self.get_reordered(new_order)



