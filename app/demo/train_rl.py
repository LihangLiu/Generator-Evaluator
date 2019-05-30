"""
train:
    train uni-rnn to predict:
        - 'click': the ctr
        - 'click_credit': the ctr and the credit
test:
    test the auc and the rmse
generate_list:
    generate list by uni-rnn
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import sys
import copy
import time
import datetime
import collections
import os
from os.path import basename, join, exists, dirname
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # filename='hot_rl.log', 

#########
# envs
#########

import tensorflow as tf

import paddle
from paddle import fluid

from config import Config
from utils import (sequence_unconcat, sequence_expand, sequence_gather, sequence_sampling, 
                    BatchData, add_scalar_summary)

import _init_paths

from src.eval_net import BiRNN, Transformer
from src.rl_net import RLUniRNN
from src.gen_algorithm import GenAlgorithm
from src.rl_algorithm import RLAlgorithm
from src.gen_computation_task import GenComputationTask

from src.utils import (read_json, print_args, tik, tok, save_pickle, threaded_generator, 
                        AUCMetrics, AssertEqual)
from src.fluid_utils import (fluid_create_lod_tensor as create_tensor, 
                            concat_list_array, seq_len_2_lod, get_num_devices)
from data.npz_dataset import NpzDataset, FakeTensor

from train_sl import main as eval_entry_func
from train_sl import SLFeedConvertor

#########
# utils
#########

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', help="Exp id, used for logs or savings")
    parser.add_argument('--use_cuda', default = 1, type = int, help = "")
    parser.add_argument('--train_mode', 
                        default = 'single', 
                        choices = ['single', 'parallel'],
                        type = str, 
                        help = "single: use the first gpu, parallel: use all gpus")
    parser.add_argument('--task', 
                        default = 'train', 
                        choices = ['train', 'test', 'debug', 'eval_list'],
                        type = str, 
                        help = "")

    # model settings
    parser.add_argument('--gamma', type=float, help='')
    parser.add_argument('--eval_exp', type=str, help='')

    # eval_list settings
    parser.add_argument('--eval_npz_list', type=str, default='', help='')
    return parser


class RLFeedConvertor(object):
    @staticmethod
    def infer_init(batch_data):
        place = fluid.CPUPlace()
        feed_dict = {}
        for name in batch_data.conf.user_slot_names:
            ft = batch_data.tensor_dict[name]
            feed_dict[name] = create_tensor(ft.values, lod=ft.lod, place=place)
        return feed_dict

    @staticmethod
    def infer_onestep(batch_data, step_id, prev_hidden, candidate_items):
        """
        prev_hidden: len() = batch_size, e.g. [(dim), [], (dim), ...]
        candidate_items: len() = batch_size, e.g. [[2,3,6], [], [4,3], ...]
        """
        cand_lens = [len(ci) for ci in candidate_items]

        global_item_indice = []
        for ci, offset in zip(candidate_items, batch_data.offset()):
            if len(ci) > 0:
                global_item_indice.append(ci + offset)
        global_item_indice = np.concatenate(global_item_indice, axis=0)

        place = fluid.CPUPlace()
        feed_dict = {}
        input_item_lod = [seq_len_2_lod([1] * np.sum(cand_lens))]
        for name in batch_data.conf.item_slot_names:
            v = batch_data.tensor_dict[name].values
            v = v[global_item_indice]
            feed_dict[name] = create_tensor(v, lod=input_item_lod, place=place)

        input_pos = np.array([step_id] * np.sum(cand_lens)).reshape([-1, 1])
        feed_dict['pos'] = create_tensor(input_pos, lod=input_item_lod, place=place)
        input_hidden = sequence_expand(prev_hidden, cand_lens)
        feed_dict['prev_hidden'] = create_tensor(input_hidden, lod=input_item_lod, place=place)
        return feed_dict

    @staticmethod
    def train_test(batch_data, reward):
        place = fluid.CPUPlace()
        feed_dict = {}
        for name in batch_data.conf.user_slot_names + batch_data.conf.item_slot_names:
            ft = batch_data.tensor_dict[name]
            feed_dict[name] = create_tensor(ft.values, lod=ft.lod, place=place)

        pos = batch_data.pos().reshape([-1, 1]).astype('int64')
        feed_dict['pos'] = create_tensor(pos, lod=batch_data.lod(), place=place)
        decode_len = batch_data.decode_len().reshape([-1, 1]).astype('int64')
        lod = [seq_len_2_lod([1] * len(decode_len))]
        feed_dict['decode_len'] = create_tensor(decode_len, lod=lod, place=place)
        reward = reward.reshape([-1, 1]).astype('float32')
        lod = [seq_len_2_lod(batch_data.decode_len())]
        feed_dict['reward'] = create_tensor(reward, lod=lod, place=place)
        return feed_dict


############
# main
############

def main(args):
    conf = Config(args.exp)

    ### build model
    npz_config = read_json(conf.npz_config_path)
    scope = fluid.Scope()
    with fluid.scope_guard(scope):
        with fluid.unique_name.guard():
            model = RLUniRNN(conf, npz_config)
            algorithm = RLAlgorithm(model,
                                    optimizer=conf.optimizer, lr=conf.lr,
                                    gpu_id=(0 if args.use_cuda == 1 else -1),
                                    gamma=args.gamma)
    td_ct = GenComputationTask(algorithm, model_dir=conf.model_dir, mode=args.train_mode, scope=scope)

    # get eval model
    eval_args = copy.deepcopy(args)
    eval_args.exp = args.eval_exp
    eval_args.model = 'BiRNN'
    eval_args.task = 'eval'
    eval_td_ct = eval_entry_func(eval_args)

    ### other tasks
    if args.task == 'test':
        test(td_ct, args, conf, None, td_ct.ckp_step)
        exit()
    elif args.task == 'eval_list':
        return eval_list(td_ct, args, conf, td_ct.ckp_step, args.eval_npz_list)
        exit()

    ### start training
    memory_size = 1000
    replay_memory = collections.deque(maxlen=memory_size)
    summary_writer = tf.summary.FileWriter(conf.summary_dir)
    for epoch_id in range(td_ct.ckp_step + 1, conf.max_train_steps):
        train(td_ct, eval_td_ct, args, conf, summary_writer, replay_memory, epoch_id)
        td_ct.save_model(conf.model_dir, epoch_id)
        test(td_ct, args, conf, summary_writer, epoch_id)


def inference(td_ct, batch_data):
    fetch_dict = td_ct.infer_init(RLFeedConvertor.infer_init(batch_data))
    prev_hidden = np.array(fetch_dict['init_hidden'])
    pre_items = [[] for _ in range(batch_data.batch_size())]
    for step_id in range(np.max(batch_data.decode_len())):
        ### evaluate candidates
        candidate_items = batch_data.get_candidates(pre_items, stop_flags=(step_id >= batch_data.decode_len()))
        cand_lens = [len(ci) for ci in candidate_items]
        feed_dict = RLFeedConvertor.infer_onestep(batch_data, step_id, prev_hidden, candidate_items)
        fetch_dict = td_ct.infer_onestep(feed_dict)
        c_Q = np.array(fetch_dict['c_Q']).flatten()         # (b*cand_len,)
        next_hidden = np.array(fetch_dict['next_hidden'])   # (b*cand_len, dim)

        ### sampling
        selected_index = sequence_sampling(c_Q, cand_lens, sampling_type='greedy')   # e.g. [1, None, 4, ...] with len = b

        ### update pre_items and prev_hidden
        for pre, si, ci in zip(pre_items, selected_index, candidate_items):
            if not si is None:
                pre.append(ci[si])
        prev_hidden = sequence_gather(next_hidden, cand_lens, selected_index)
    return pre_items


def train(td_ct, eval_td_ct, args, conf, summary_writer, replay_memory, epoch_id):
    """train for conf.train_interval steps"""
    np.random.seed(0)
    dataset = NpzDataset(conf.train_npz_list, conf.npz_config_path, conf.requested_npz_names, if_random_shuffle=True)
    data_gen = dataset.get_data_generator(conf.batch_size)

    list_loss = []
    list_steps = range(5)
    dict_c_Q = {s:[] for s in list_steps}
    batch_id = 0
    last_batch_data = BatchData(conf, data_gen.next())
    for tensor_dict in data_gen:
        ### sampling
        tik()
        batch_data = BatchData(conf, tensor_dict)
        batch_data.set_decode_len(batch_data.seq_lens())
        batch_data.expand_candidates(last_batch_data, batch_data.seq_lens())

        order = inference(td_ct, batch_data)
        tok('sampling')

        ### get reward
        tik()
        reordered_batch_data = batch_data.get_reordered(order)
        fetch_dict = eval_td_ct.inference(SLFeedConvertor.inference(reordered_batch_data))
        reward = np.array(fetch_dict['click_prob'])[:, 1]
        tok('get reward')

        ### save to replay_memory
        tik()
        reordered_batch_data2 = batch_data.get_reordered_keep_candidate(order)
        reordered_batch_data2.set_decode_len(batch_data.decode_len())
        replay_memory.append((reordered_batch_data2, reward))
        tok('replay memory')

        ### train
        tik()
        memory_batch_data, reward = replay_memory[np.random.randint(len(replay_memory))]
        feed_dict = RLFeedConvertor.train_test(memory_batch_data, reward)
        fetch_dict = td_ct.train(feed_dict)
        tok('train')

        ### logging
        list_loss.append(np.array(fetch_dict['loss']))
        c_Q = np.array(fetch_dict['c_Q'])
        start_index = np.array(fetch_dict['c_Q'].lod()[0][:-1])
        for s in list_steps:
            dict_c_Q[s].append(np.mean(c_Q[start_index + s]))
        if batch_id % conf.prt_interval == 0:
            logging.info('train/loss %f' % np.mean(list_loss))
            for s in list_steps:
                logging.info('train/c_Q_%d %f' % (s, np.mean(dict_c_Q[s])))
            list_loss = []
            dict_c_Q = {s:[] for s in list_steps}

        last_batch_data = BatchData(conf, tensor_dict)
        batch_id += 1


def test(td_ct, args, conf, summary_writer, epoch_id):
    """eval auc on the full test dataset"""
    dataset = NpzDataset(args.test_npz_list, 
                        conf.npz_config_path, 
                        conf.requested_npz_names,
                        if_random_shuffle=False)
    data_manager = DataManager(conf, dataset, conf.batch_size)

    list_loss = []
    list_steps = range(5)
    dict_c_Q = {s:[] for s in list_steps}
    batch_id = 0
    while True:
        list_batch_data, list_feed_dict = data_manager.forward()
        if list_batch_data is None:
            break
        fetch_dict = td_ct.test(list_feed_dict)

        ### logging
        list_loss.append(np.array(fetch_dict['loss']))
        c_Q = np.array(fetch_dict['c_Q'])
        start_index = np.array(fetch_dict['c_Q'].lod()[0][:-1])
        for s in list_steps:
            dict_c_Q[s].append(np.mean(c_Q[start_index + s]))

        if batch_id % 100 == 0:
            logging.info('%d loss = %.5f' % (batch_id, np.mean(list_loss)))
        batch_id += 1

    add_scalar_summary(summary_writer, epoch_id, 'test/loss', np.mean(list_loss))
    for s in list_steps:
        add_scalar_summary(summary_writer, epoch_id, 'test/c_Q_%d' % s, np.mean(dict_c_Q[s]))


def eval_list(td_ct, args, conf, npz_list):
    """
    return unshuffled npz_list dataset and td_ct
    """
    dataset = NpzDataset(npz_list, 
                        conf.npz_config_path, 
                        conf.requested_npz_names,
                        if_random_shuffle=False)
    print ('load', npz_list)
    data_manager = DataManager(conf, dataset, conf.batch_size)
    return data_manager, td_ct

    

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)


