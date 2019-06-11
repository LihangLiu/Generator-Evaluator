"""
    Model Components
"""
from __future__ import print_function
import sys
import traceback
import numpy as np
from threading import Thread
import math
import pickle
import json
import time
import thread
import hashlib
from Queue import Queue
from collections import OrderedDict
import os
from os.path import exists, dirname
from sklearn.metrics import roc_auc_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class TracebackWrapper:
    """
    eg:
        a = some_func(some_input)
        # convert to
        a = TracebackWrapper(some_func)(some_input)
    """
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **argvs):
        try:
            return self.func(*args, **argvs)
        except KeyboardInterrupt:
            print ('KeyboardInterrupt triggered.')
            exit()
        except Exception as e:
            # This prints the type, value, and stack trace of the
            # current exception being handled.
            traceback.print_exc()
            print ('Input is: %s' % str(args)[:10000])
            raise e


class OnlineRule:
    def __init__(self, feature_id_map_reverse, rule_type):
        """
        'fixtop3', 'finsert', 'video'
        """
        self.feature_id_map_reverse = feature_id_map_reverse
        self.rule_type = rule_type
        assert rule_type in ['fixtop3_finsert_video', 'finsert_video'], (rule_type)

    def _get_force_insert(self, queue_id):
        return queue_id == self.feature_id_map_reverse['queue_id']['52']

    def _get_video(self, mark_id, queue_id):
        res = (mark_id == self.feature_id_map_reverse['mark_id']['6'])
        res = np.logical_or(res, mark_id == self.feature_id_map_reverse['mark_id']['7'])
        res = np.logical_or(res, queue_id == self.feature_id_map_reverse['queue_id']['9'])
        res = np.logical_or(res, queue_id == self.feature_id_map_reverse['queue_id']['106'])
        return res

    def check(self, order, mark_id, queue_id):
        """
        batch version.

        args:
            mark_id: (batch, seq_len), global
            queue_id: (batch, seq_len), global
            order: 2d-list of real seq_len
        """
        if_force_insert = self._get_force_insert(queue_id)
        if_video = self._get_video(mark_id, queue_id)
        batch_size = len(order)
        for b in range(batch_size):
            c_f_insert = if_force_insert[b]
            c_if_video = if_video[b]
            c_order = order[b]

            print ('------', b, '-------')
            print ('force insert', c_f_insert.astype(int)[:len(c_order)])
            print ('order', c_order)
            print ('video', np.array([c_if_video[x] for x in c_order]).astype(int))

    def apply(self, prev_ids, mark_id, queue_id, prev_mask, step):
        """
        batch version.
        fix top 3, force insert and video discountinuous.

        args:
            prev_ids: (batch, ), previous id, starting from 0. 
                    Special treat to -1
            mark_id: (batch, max_seq_len), global
            queue_id: (batch, max_seq_len), global
            prev_mask: (batch, max_seq_len), 1: invalid, 0: valid
            step: int, starting from 0
        returns:
            mask: (batch, seq_len), 1: invalid, 0: valid
        """
        ### TODO, cut shape due to replay memory
        batch_size, max_seq_len = prev_mask.shape
        mark_id = mark_id[:, :max_seq_len]
        queue_id = queue_id[:, :max_seq_len]
        assert prev_mask.shape == mark_id.shape and prev_mask.shape == queue_id.shape, (prev_mask.shape,
                                                                                        mark_id.shape,
                                                                                        queue_id.shape)

        ### fix top 3
        if 'fixtop3' in self.rule_type:
            if step < 3:
                mask = np.ones_like(prev_mask, np.float32)
                mask[:, step] = 0
                return mask

        ### force insert and video
        if_force_insert = self._get_force_insert(queue_id)
        if_video = self._get_video(mark_id, queue_id)
        mask = []
        for b in range(batch_size):
            c_f_insert = if_force_insert[b]
            c_last_choice = prev_ids[b]
            c_if_video = if_video[b]
            c_prev_mask = prev_mask[b]
            # if force insert
            if c_f_insert[step] == 1:
                c_mask = np.ones([max_seq_len])
                c_mask[step] = 0
            else:
                c_mask = np.zeros([max_seq_len])
                c_mask[c_prev_mask == 1] = 1    # exclude previously selected
                c_mask[c_f_insert == 1] = 1     # exclude force insert
                # if last choice is video
                c_tmp_mask = np.array(c_mask)
                if c_last_choice >= 0 and c_if_video[c_last_choice] == 1:
                    c_tmp_mask[c_if_video == 1] = 1
                # if leaves nothing, remove video constraints
                if not (c_tmp_mask == 1).all():
                    c_mask = c_tmp_mask
            mask.append(c_mask)

        mask = np.array(mask, dtype=prev_mask.dtype)
        return mask

class BiasRewardEval:
    def __init__(self, feature_id_map):
        self.feature_id_map = feature_id_map
        self.queue_id_map = self.feature_id_map['queue_id']

    @property
    def bias_dict(self):
        """
        queue
        """
        bias_dict = {
            '106': 60,
            '15': 1,
            '38': 30,
            '17': 45,
            '34': 22,
            '65': 15,
            '14': 5,
            '33': 23,
            '16': 5,
        }
        return bias_dict
    
    def get_reward(self, queue_id, order):
        """
        batch_version.

        args:
            queue_id: (batch, max_seq_len)
            order: 2d list with real seq_len

        """
        reward = []
        batch_size = len(order)
        for b in range(batch_size):
            c_ordered_queue_id = [queue_id[b][x] for x in order[b]]
            c_reward = []
            for x in c_ordered_queue_id:
                if x in self.queue_id_map and self.queue_id_map[x] in self.bias_dict:
                    c_reward.append(self.bias_dict[self.queue_id_map[x]])
                else:
                    c_reward.append(0)
            reward.append(c_reward)
        return reward


class AUCMetrics(object):
    """docstring for AUCMetrics"""
    def __init__(self):
        self.labels = []
        self.y_scores = []
        self.cuids = []
        self.seq_lens = []

    def _sequence_expand(self, vec1d, seq_len):
        assert len(vec1d) == len(seq_len)
        exp_vec1d = []
        for e, s in zip(vec1d, seq_len):
            exp_vec1d += [e] * s
        return np.array(exp_vec1d)

    def add(self, labels, y_scores, cuids=None, seq_lens=None):
        """
        batch add
        cuids and seq_lens is necessary for cuid_wise_auc
        """
        AssertEqual(len(labels), len(y_scores))
        self.labels += list(labels)
        self.y_scores += list(y_scores)
        if not cuids is None:
            AssertEqual(len(labels), np.sum(seq_lens))
            AssertEqual(len(seq_lens), len(cuids))
            self.cuids += list(cuids)
            self.seq_lens += list(seq_lens)

    def num_items(self):
        """the overall num of items"""
        return len(self.labels)

    def E_label(self):
        return np.mean(self.labels)

    def E_pred(self):
        return np.mean(self.y_scores)

    def overall_auc(self):
        """return the auc of all items as a whole"""
        return roc_auc_score(self.labels, self.y_scores)

    def cuid_wise_auc(self):
        """return the mean auc of all cuids"""
        if len(self.cuids) == 0:
            return 0, 0, 0

        cuid_expanded = self._sequence_expand(self.cuids, self.seq_lens)
        cuid_dict = {}
        for c, l, y in zip(cuid_expanded, self.labels, self.y_scores):
            if c not in cuid_dict:
                cuid_dict[c] = {'label':[], 'y_score':[]}
            cuid_dict[c]['label'].append(l)
            cuid_dict[c]['y_score'].append(y)
        
        list_auc = []
        for key in cuid_dict:
            if len(set(cuid_dict[key]['label'])) <= 1:
                continue
            list_auc.append(roc_auc_score(cuid_dict[key]['label'], cuid_dict[key]['y_score']))
        cuid_wise_auc = np.mean(list_auc)
        num_valid_cuids = len(list_auc)
        num_cuids = len(cuid_dict)
        return cuid_wise_auc, num_valid_cuids, num_cuids

    def dump(self, filename):
        """dump related variables to the npz file"""
        if not exists(dirname(filename)) and dirname(filename) != '':
            os.makedirs(dirname(filename))
        np.savez_compressed(filename, 
                            labels=self.labels,
                            y_scores=self.y_scores,
                            cuids=self.cuids,
                            seq_lens=self.seq_lens)
        print ('Saved file to %s' % filename)
        

class AccuracyMetrics(object):
    """docstring for AccuracyMetrics"""
    def __init__(self):
        self.labels = []
        self.pred = []
        self.n_class = None

        self.correct_count = 0
        self.total_count = 0

    def add(self, labels, probs):
        """
        batch add
        """
        labels = np.array(labels)
        probs = np.array(probs)
        AssertEqual(len(labels), len(probs))
        AssertLT(np.max(labels), probs.shape[1])
        if self.n_class is None:
            self.n_class = probs.shape[1]
        AssertEqual(probs.shape[1], self.n_class)

        pred = np.argmax(probs, 1)
        self.labels += list(labels)
        self.pred += list(pred)

        self.correct_count += np.sum(pred == labels)
        self.total_count += len(labels)
    
    def overall_accuracy(self):
        return self.correct_count / float(self.total_count)

    def class_accuracy(self):
        od = OrderedDict()
        pred = np.array(self.pred)
        labels = np.array(self.labels)
        for class_i in range(self.n_class):
            index = (labels == class_i)
            num = np.sum(index)
            if num == 0:
                continue
            accuracy = np.mean(pred[index] == labels[index])
            od[class_i] = {'num':num, 'accu':round(accuracy, 3)}
        return od


class RMSEMetrics(object):
    """docstring for AccuracyMetrics"""
    def __init__(self):
        self.total_mse = 0.0
        self.total_count = 0

    def add(self, labels, preds):
        """
        batch add
        """
        labels = np.array(labels)
        preds = np.array(preds)
        AssertEqual(labels.shape, preds.shape)
        AssertEqual(len(labels.shape), 1)

        self.total_mse += np.sum((labels - preds) ** 2)
        self.total_count += len(labels)
    
    def overall_rmse(self):
        return np.sqrt(self.total_mse / self.total_count)


class SequenceRMSEMetrics(object):
    """docstring for AccuracyMetrics"""
    def __init__(self):
        self.total_mse = 0.0
        self.total_count = 0

    def add(self, labels, preds):
        """
        add one sequence a time
        """
        labels = np.array(labels)
        preds = np.array(preds)
        AssertEqual(labels.shape, preds.shape)
        AssertEqual(len(labels.shape), 1)

        self.total_mse += (np.sum(labels) - np.sum(preds)) ** 2
        self.total_count += len(labels)
    
    def overall_rmse(self):
        return np.sqrt(self.total_mse / self.total_count)


class CppInputSlot(object):
    """Define format of an input feature for GR code"""
    def __init__(self, 
                category=None, 
                input_name=None, 
                field_name=None, 
                input_type=None, 
                shapes=None, 
                lod_level=None):
        self.category = category
        self.input_name = input_name
        self.field_name = field_name
        self.input_type = input_type
        self.shapes = shapes
        self.lod_level = lod_level
    
    def __str__(self):
        shape_str = '\n'.join(['    shape: %d' % s for s in self.shapes])
        body = """
{category} {{
    input_name: "{input_name}"
    field_name: "{field_name}"
    input_type: "{input_type}"
{shape_str}
    lod_level: {lod_level}
}}
""" 
        body = body.format(shape_str=shape_str, **self.__dict__)
        return body.strip()


class CppOutputSlot(object):
    """Define format of an output feature for GR code"""
    def __init__(self, 
                category=None, 
                output_name=None, 
                output_type=None, 
                shapes=None, 
                lod_level=None):
        self.category = category
        self.output_name = output_name
        self.output_type = output_type
        self.shapes = shapes
        self.lod_level = lod_level
    
    def __str__(self):
        shape_str = '\n'.join(['    shape: %d' % s for s in self.shapes])
        body = """
{category} {{
    output_name: "{output_name}"
    output_type: "{output_type}"
{shape_str}
    lod_level: {lod_level}
}}
""" 
        body = body.format(shape_str=shape_str, **self.__dict__)
        return body.strip()


class CppInputProfile(object):
    """Define format of features of a network for GR code"""
    def __init__(self):
        self.slots = []

    def add_slot(self, slot):
        """add a input/output slot in order"""
        self.slots.append(slot)

    def __str__(self):
        return '\n'.join([str(s) for s in self.slots])


tik_tok_stack = []
def tik():
    tik_tok_stack.append(time.time())
def tok(message='Time used:'):
    last_time = tik_tok_stack.pop()
    logging.info ('%s %.4fs' % (message, time.time() - last_time))

def print_args(args, message=''):
    print (message)
    for key in vars(args):
        print('    [{0}] = {1}'.format(key, getattr(args, key)))

set_print_once = set()
def print_once(message, id=None):
    """use message as id if id is None"""
    if id is None:
        id = message
    if id in set_print_once:
        return
    logging.info(message)
    set_print_once.add(id)

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def read_pickle(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

def save_pickle(filename, var):
    if not exists(dirname(filename)) and dirname(filename) != '':
        os.makedirs(dirname(filename))
    with open(filename, 'wb') as handle:
        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info('saved to %s' % filename)

def save_json(file, dict_obj):
    if not exists(dirname(file)) and dirname(file) != '':
        os.makedirs(dirname(file))
    with open(file, 'w') as outfile:  
        json.dump(dict_obj, outfile)
    logging.info('saved to %s' % file)

def read_json(file):
    with open(file, 'r') as json_file:  
        data = json.load(json_file)
    return data

def save_lines_2_txt(file, lines):
    if not exists(dirname(file)) and dirname(file) != '':
        os.makedirs(dirname(file))
    with open(file, 'w') as outfile:  
        for line in lines:
            outfile.write(line + '\n')
    logging.info('saved to %s' % file)

def seq_len_2_lod(list_seq_len):
    return [0] + list(np.cumsum(list_seq_len))

def read_feature_id_map(list_feature_id_map):
    feature_id_map = {}
    feature_id_map_reverse = {}
    with open(list_feature_id_map, 'r') as f:
        for line in f.readlines():
            if not '=' in line:
                continue

            segs = line.strip().split('\t')
            try:
                feature_name = segs[0][:segs[0].find('=')]
                score_value = segs[0][segs[0].find('=') + 1:]
                # feature_name, score_value = segs[0].split('=')
                score_id = int(segs[1])
            except:
                print ('line', line)
                print ('segs', segs)
                exit()

            # add '_id'
            feature_name += '_id'

            if not feature_name in feature_id_map:
                feature_id_map[feature_name] = {}
                feature_id_map_reverse[feature_name] = {}
            feature_id_map[feature_name][score_id] = score_value   # key: int, value: str
            feature_id_map_reverse[feature_name][score_value] = score_id   # key: str, value: int
        return feature_id_map, feature_id_map_reverse


def get_boundaries_and_values_for_piecewise_decay(init_lr, warmup_proportion, 
                                                decay_rate, num_decays, decay_steps, start_step=0):
    """
    Used for fluid.layers.piecewise_decay
    """
    boundaries, values = [], []

    # warmup part: init_lr * current_step / num_warmup_steps
    if warmup_proportion == 0:
        boundaries.append(0)
        values.append(init_lr)
    else:
        num_warmup_steps = int(decay_steps * warmup_proportion)
        warmup_boundaries = np.linspace(0, num_warmup_steps, num=20, endpoint=True).astype(int)
        warmup_values = init_lr * warmup_boundaries / num_warmup_steps
        warmup_values[0] = warmup_values[1]
        boundaries += list(warmup_boundaries)
        values += list(warmup_values)

    # decay part: init_lr * decay_rate ** i
    for i in range(num_decays):
        boundaries.append(int(decay_steps * (i + 1)))
        values.append(init_lr * decay_rate ** (i + 1))

    del boundaries[0]
    assert len(set(boundaries)) == len(boundaries), (boundaries)
    return boundaries, values


def threaded_generator(gen, capacity):
    """
    threading wrapper
    Use None as EOF
    """
    queue = Queue(capacity)

    def thread_push():
        """push mix_data to the queue"""
        for element in gen:
            queue.put(element)
        queue.put(None)     # mark as EOF
    thread.start_new_thread(thread_push, ())

    while True:
        element = queue.get()
        if element is None:
            yield None   
            break
        yield element

def AssertEqual(e1, e2, message=""):
    assert e1 == e2, (message, e1, e2)

def AssertLT(e1, e2, message=""):
    assert e1 < e2, (message, e1, e2)




