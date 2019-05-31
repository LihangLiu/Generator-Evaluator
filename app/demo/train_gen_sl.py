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

from src.gen_net import DNN, UniRNN
from src.gen_algorithm import GenAlgorithm
from src.gen_computation_task import GenComputationTask

from src.utils import (read_json, print_args, tik, tok, save_pickle, threaded_generator, 
                        AUCMetrics, AssertEqual)
from src.fluid_utils import (fluid_create_lod_tensor as create_tensor, 
                            concat_list_array, seq_len_2_lod, get_num_devices)
from data.npz_dataset import NpzDataset

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
                        choices = ['train', 'test', 'sampling', 'debug'],
                        type = str, 
                        help = "")
    
    # model settings
    parser.add_argument('--model', type=str, choices=['DNN', 'UniRNN'], help='')
    parser.add_argument('--eval_exp', type=str, help='')
    return parser


class GenSLFeedConvertor(object):
    @staticmethod
    def train_test(batch_data):
        place = fluid.CPUPlace()
        feed_dict = {}
        for name in batch_data.conf.user_slot_names + batch_data.conf.item_slot_names:
            ft = batch_data.tensor_dict[name]
            feed_dict[name] = create_tensor(ft.values, lod=ft.lod, place=place)

        click_id = batch_data.tensor_dict['click_id']
        feed_dict['click_id'] = create_tensor(click_id.values, lod=click_id.lod, place=place)
        return feed_dict

    @staticmethod
    def inference(batch_data):
        place = fluid.CPUPlace()
        feed_dict = {}
        for name in batch_data.conf.user_slot_names + batch_data.conf.item_slot_names:
            ft = batch_data.tensor_dict[name]
            feed_dict[name] = create_tensor(ft.values, lod=ft.lod, place=place)            
        return feed_dict

    @staticmethod
    def sampling(batch_data):
        place = fluid.CPUPlace()
        feed_dict = {}
        for name in batch_data.conf.user_slot_names + batch_data.conf.item_slot_names:
            ft = batch_data.tensor_dict[name]
            feed_dict[name] = create_tensor(ft.values, lod=ft.lod, place=place)

        decode_len = batch_data.decode_len().reshape([-1, 1]).astype('int64')
        lod = [seq_len_2_lod([1] * len(decode_len))]
        feed_dict['decode_len'] = create_tensor(decode_len, lod=lod, place=place)
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
            if args.model == 'DNN':
                model = DNN(conf, npz_config)
            elif args.model == 'UniRNN':
                model = UniRNN(conf, npz_config)

            algorithm = GenAlgorithm(model, optimizer=conf.optimizer, lr=conf.lr, gpu_id=(0 if args.use_cuda == 1 else -1))
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
    elif args.task == 'sampling':
        sampling(td_ct, eval_td_ct, args, conf, None, td_ct.ckp_step)
        exit()

    ### start training
    summary_writer = tf.summary.FileWriter(conf.summary_dir)
    for epoch_id in range(td_ct.ckp_step + 1, conf.max_train_steps):
        train(td_ct, args, conf, summary_writer, epoch_id)
        td_ct.save_model(conf.model_dir, epoch_id)
        test(td_ct, args, conf, summary_writer, epoch_id)


def train(td_ct, args, conf, summary_writer, epoch_id):
    """train for conf.train_interval steps"""
    dataset = NpzDataset(conf.train_npz_list, conf.npz_config_path, conf.requested_npz_names, if_random_shuffle=True)
    data_gen = dataset.get_data_generator(conf.batch_size)

    list_epoch_loss = []
    list_loss = []
    batch_id = 0
    for tensor_dict in data_gen:
        batch_data = BatchData(conf, tensor_dict)
        fetch_dict = td_ct.train(GenSLFeedConvertor.train_test(batch_data))
        list_loss.append(np.array(fetch_dict['loss']))
        list_epoch_loss.append(np.mean(np.array(fetch_dict['loss'])))
        if batch_id % conf.prt_interval == 0:
            logging.info('batch_id:%d loss:%f' % (batch_id, np.mean(list_loss)))
            list_loss = []
        batch_id += 1

    add_scalar_summary(summary_writer, epoch_id, 'train/loss', np.mean(list_epoch_loss))


def test(td_ct, args, conf, summary_writer, epoch_id):
    """eval auc on the full test dataset"""
    dataset = NpzDataset(conf.test_npz_list, conf.npz_config_path, conf.requested_npz_names, if_random_shuffle=False)
    data_gen = dataset.get_data_generator(conf.batch_size)

    auc_metric = AUCMetrics()
    batch_id = 0
    for tensor_dict in data_gen:
        batch_data = BatchData(conf, tensor_dict)
        fetch_dict = td_ct.test(GenSLFeedConvertor.train_test(batch_data))
        auc_metric.add(labels=np.array(fetch_dict['click_id']).flatten(),
                        y_scores=np.array(fetch_dict['click_prob'])[:, 1])
        batch_id += 1

    add_scalar_summary(summary_writer, epoch_id, 'test/auc', auc_metric.overall_auc())


def sampling(td_ct, eval_td_ct, args, conf, summary_writer, epoch_id):
    """sampling"""
    dataset = NpzDataset(conf.test_npz_list, conf.npz_config_path, conf.requested_npz_names, if_random_shuffle=False)
    data_gen = dataset.get_data_generator(conf.batch_size)

    list_reward = []
    last_batch_data = BatchData(conf, data_gen.next())
    for tensor_dict in data_gen:
        ### sampling
        batch_data = BatchData(conf, tensor_dict)
        batch_data.set_decode_len(batch_data.seq_lens())
        batch_data.expand_candidates(last_batch_data, batch_data.seq_lens())

        fetch_dict = td_ct.sampling(GenSLFeedConvertor.sampling(batch_data))
        sampled_id = np.array(fetch_dict['sampled_id']).reshape([-1])
        order = sequence_unconcat(sampled_id, batch_data.decode_len())

        ### get reward
        reordered_batch_data = batch_data.get_reordered(order)
        fetch_dict = eval_td_ct.inference(EvalFeedConvertor.inference(reordered_batch_data))
        reward = np.array(fetch_dict['click_prob'])[:, 1]

        ### logging
        list_reward.append(np.mean(reward))
        print('reward', reward.shape, np.mean(reward))

        last_batch_data = BatchData(conf, tensor_dict)

    add_scalar_summary(summary_writer, global_batch_id, 'sampling/reward', np.mean(list_reward))

    

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)


