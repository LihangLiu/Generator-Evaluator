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
from src.rl_net import RLUniRNN, RLPointerNet
from src.gen_algorithm import GenAlgorithm
from src.rl_algorithm import RLAlgorithm
from src.rl_computation_task import RLComputationTask

from src.utils import (read_json, print_args, tik, tok, save_pickle, threaded_generator, 
                        AUCMetrics, AssertEqual)
from src.fluid_utils import (fluid_create_lod_tensor as create_tensor, 
                            concat_list_array, seq_len_2_lod, get_num_devices)
from data.npz_dataset import NpzDataset, FakeTensor

from train_eval import main as eval_entry_func
from train_eval import EvalFeedConvertor

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
                        choices = ['train', 'sampling', 'debug'],
                        type = str, 
                        help = "")

    # model settings
    parser.add_argument('--model', choices=['UniRNN', 'PointerNet'], help='')
    parser.add_argument('--gamma', type=float, help='')
    parser.add_argument('--candidate_encode', help='')
    parser.add_argument('--attention_type', default='dot', help='')
    parser.add_argument('--log_reward', type=int, default=0, help='')
    parser.add_argument('--eval_exp', type=str, help='')
    parser.add_argument('--eval_model', type=str, help='')
    return parser


class GenRLFeedConvertor(object):
    @staticmethod
    def train_test(batch_data, reward):
        place = fluid.CPUPlace()
        feed_dict = {}
        for name in batch_data.conf.user_slot_names + batch_data.conf.item_slot_names:
            ft = batch_data.tensor_dict[name]
            feed_dict[name] = create_tensor(ft.values, lod=ft.lod, place=place)

        decode_len = batch_data.decode_len().reshape([-1, 1]).astype('int64')
        lod = [seq_len_2_lod([1] * len(decode_len))]
        feed_dict['decode_len'] = create_tensor(decode_len, lod=lod, place=place)
        reward = reward.reshape([-1, 1]).astype('float32')
        lod = [seq_len_2_lod(batch_data.decode_len())]
        feed_dict['reward'] = create_tensor(reward, lod=lod, place=place)
        return feed_dict

    @staticmethod
    def sampling(batch_data, eps):
        place = fluid.CPUPlace()
        feed_dict = {}
        for name in batch_data.conf.user_slot_names + batch_data.conf.item_slot_names:
            ft = batch_data.tensor_dict[name]
            feed_dict[name] = create_tensor(ft.values, lod=ft.lod, place=place)

        decode_len = batch_data.decode_len().reshape([-1, 1]).astype('int64')
        lod = [seq_len_2_lod([1] * len(decode_len))]
        feed_dict['decode_len'] = create_tensor(decode_len, lod=lod, place=place)
        eps = np.array([eps]).astype('float32')
        feed_dict['eps'] = create_tensor(eps, lod=[], place=place)
        return feed_dict


############
# main
############

def main(args):
    print_args(args, 'args')
    conf = Config(args.exp)

    ### build model
    npz_config = read_json(conf.npz_config_path)
    scope = fluid.Scope()
    with fluid.scope_guard(scope):
        with fluid.unique_name.guard():
            if args.model == 'UniRNN':
                model = RLUniRNN(conf, npz_config, candidate_encode=args.candidate_encode)
            elif args.model == 'PointerNet':
                model = RLPointerNet(conf, npz_config, candidate_encode=args.candidate_encode)
            algorithm = RLAlgorithm(model,
                                    optimizer=conf.optimizer, lr=conf.lr,
                                    gpu_id=(0 if args.use_cuda == 1 else -1),
                                    gamma=args.gamma)
    td_ct = RLComputationTask(algorithm, model_dir=conf.model_dir, mode=args.train_mode, scope=scope)

    # get eval model
    eval_args = copy.deepcopy(args)
    eval_args.exp = args.eval_exp
    eval_args.model = args.eval_model
    eval_args.task = 'eval'
    eval_td_ct = eval_entry_func(eval_args)

    ### other tasks
    if args.task == 'sampling':
        sampling(td_ct, eval_td_ct, args, conf, None, td_ct.ckp_step)
        exit()

    ### start training
    memory_size = 1000
    replay_memory = collections.deque(maxlen=memory_size)
    summary_writer = tf.summary.FileWriter(conf.summary_dir)
    for epoch_id in range(td_ct.ckp_step + 1, conf.max_train_steps):
        if args.log_reward == 1:
            log_train(td_ct, args, conf, summary_writer, replay_memory, epoch_id)
        else:
            train(td_ct, eval_td_ct, args, conf, summary_writer, replay_memory, epoch_id)
        td_ct.save_model(conf.model_dir, epoch_id)
        sampling(td_ct, eval_td_ct, args, conf, summary_writer, epoch_id)


def train(td_ct, eval_td_ct, args, conf, summary_writer, replay_memory, epoch_id):
    """train"""
    dataset = NpzDataset(conf.train_npz_list, conf.npz_config_path, conf.requested_npz_names, if_random_shuffle=True)
    data_gen = dataset.get_data_generator(conf.batch_size)

    list_reward = []
    list_loss = []
    list_first_Q = []
    guessed_batch_num = 11500
    batch_id = 0
    last_batch_data = BatchData(conf, data_gen.next())
    for tensor_dict in data_gen:
        ### sampling
        batch_data = BatchData(conf, tensor_dict)
        batch_data.set_decode_len(batch_data.seq_lens())
        batch_data.expand_candidates(last_batch_data, batch_data.seq_lens())

        fetch_dict = td_ct.sampling(GenRLFeedConvertor.sampling(batch_data, eps=0.2))
        sampled_id = np.array(fetch_dict['sampled_id']).reshape([-1])
        order = sequence_unconcat(sampled_id, batch_data.decode_len())

        ### get reward
        reordered_batch_data = batch_data.get_reordered(order)
        fetch_dict = eval_td_ct.inference(EvalFeedConvertor.inference(reordered_batch_data))
        reward = np.array(fetch_dict['click_prob'])[:, 1]

        ### save to replay_memory
        reordered_batch_data2 = batch_data.get_reordered_keep_candidate(order)
        reordered_batch_data2.set_decode_len(batch_data.decode_len())
        replay_memory.append((reordered_batch_data2, reward))

        ### train
        memory_batch_data, reward = replay_memory[np.random.randint(len(replay_memory))]
        feed_dict = GenRLFeedConvertor.train_test(memory_batch_data, reward)
        fetch_dict = td_ct.train(feed_dict)

        ### logging
        list_reward.append(np.mean(reward))
        list_loss.append(np.array(fetch_dict['loss']))
        list_first_Q.append(np.mean(np.array(fetch_dict['c_Q'])[0]))
        if batch_id % 10 == 0:
            global_batch_id = epoch_id * guessed_batch_num + batch_id
            add_scalar_summary(summary_writer, global_batch_id, 'train/rl_reward', np.mean(list_reward))
            add_scalar_summary(summary_writer, global_batch_id, 'train/rl_loss', np.mean(list_loss))
            add_scalar_summary(summary_writer, global_batch_id, 'train/rl_1st_Q', np.mean(list_first_Q))
            list_reward = []
            list_loss = []
            list_first_Q = []

        last_batch_data = BatchData(conf, tensor_dict)
        batch_id += 1


def log_train(td_ct, args, conf, summary_writer, replay_memory, epoch_id):
    """train"""
    dataset = NpzDataset(conf.train_npz_list, conf.npz_config_path, conf.requested_npz_names, if_random_shuffle=True)
    data_gen = dataset.get_data_generator(conf.batch_size)

    list_reward = []
    list_loss = []
    list_first_Q = []
    guessed_batch_num = 11500
    batch_id = 0
    last_batch_data = BatchData(conf, data_gen.next())
    for tensor_dict in data_gen:
        ### sampling
        batch_data = BatchData(conf, tensor_dict)
        batch_data.set_decode_len(batch_data.seq_lens())
        order = [np.arange(d) for d in batch_data.decode_len()]

        ### get reward
        reordered_batch_data = batch_data.get_reordered(order)
        reordered_batch_data.set_decode_len(batch_data.decode_len())
        reward = batch_data.tensor_dict['click_id'].values

        ### save to replay_memory
        replay_memory.append((reordered_batch_data, reward))

        ### train
        memory_batch_data, reward = replay_memory[np.random.randint(len(replay_memory))]
        feed_dict = GenRLFeedConvertor.train_test(memory_batch_data, reward)
        fetch_dict = td_ct.train(feed_dict)

        ### logging
        list_reward.append(np.mean(reward))
        list_loss.append(np.array(fetch_dict['loss']))
        list_first_Q.append(np.mean(np.array(fetch_dict['c_Q'])[0]))
        if batch_id % 10 == 0:
            global_batch_id = epoch_id * guessed_batch_num + batch_id
            add_scalar_summary(summary_writer, global_batch_id, 'train/rl_reward', np.mean(list_reward))
            add_scalar_summary(summary_writer, global_batch_id, 'train/rl_loss', np.mean(list_loss))
            add_scalar_summary(summary_writer, global_batch_id, 'train/rl_1st_Q', np.mean(list_first_Q))
            list_reward = []
            list_loss = []
            list_first_Q = []

        last_batch_data = BatchData(conf, tensor_dict)
        batch_id += 1


def sampling(td_ct, eval_td_ct, args, conf, summary_writer, epoch_id):
    """sampling"""
    dataset = NpzDataset(conf.test_npz_list, conf.npz_config_path, conf.requested_npz_names, if_random_shuffle=False)
    data_gen = dataset.get_data_generator(conf.batch_size)

    list_reward = []
    last_batch_data = BatchData(conf, data_gen.next())
    batch_id = 0
    for tensor_dict in data_gen:
        ### sampling
        batch_data = BatchData(conf, tensor_dict)
        batch_data.set_decode_len(batch_data.seq_lens())
        batch_data.expand_candidates(last_batch_data, batch_data.seq_lens())

        fetch_dict = td_ct.sampling(GenRLFeedConvertor.sampling(batch_data, eps=0))
        sampled_id = np.array(fetch_dict['sampled_id']).reshape([-1])
        order = sequence_unconcat(sampled_id, batch_data.decode_len())

        ### get reward
        reordered_batch_data = batch_data.get_reordered(order)
        fetch_dict = eval_td_ct.inference(EvalFeedConvertor.inference(reordered_batch_data))
        reward = np.array(fetch_dict['click_prob'])[:, 1]

        ### logging
        list_reward.append(np.mean(reward))

        if batch_id == 100:
            break

        last_batch_data = BatchData(conf, tensor_dict)
        batch_id += 1

    add_scalar_summary(summary_writer, epoch_id, 'sampling/reward-%s' % args.eval_exp, np.mean(list_reward))

    

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)


