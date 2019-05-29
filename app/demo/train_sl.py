"""
train:
    train uni-rnn to predict:
        - 'click': the ctr
        - 'click_credit': the ctr and the credit
test:
    test the auc and the mse
generate_list:
    generate list by uni-rnn
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import sys
import math
import copy
import time
import datetime
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
import paddle.fluid.profiler as profiler

from config import Config
from utils import (sequence_unconcat, sequence_expand, sequence_gather, sequence_sampling, 
                    BatchData, add_scalar_summary)

import _init_paths

from src.gen_net import DNN, UniRNN
from src.eval_net import BiRNN, Transformer
from src.gen_algorithm import GenAlgorithm
from src.eval_algorithm import EvalAlgorithm
from src.gen_computation_task import GenComputationTask
from src.eval_computation_task import EvalComputationTask

from src.utils import (read_json, print_args, tik, tok, threaded_generator, print_once,
                        AUCMetrics, AssertEqual)
from src.fluid_utils import (fluid_create_lod_tensor as create_tensor, 
                            concat_list_array, seq_len_2_lod, get_num_devices)
from data.npz_dataset import NpzDataset

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
                        choices = ['train', 'test', 'eval'],
                        type = str, 
                        help = "")
    
    # model settings
    parser.add_argument('--model', type=str, choices=['DNN', 'UniRNN', 'BiRNN', 'Trans'], help='')
    return parser


class SLFeedConvertor(object):
    @staticmethod
    def train_test(batch_data):
        place = fluid.CPUPlace()
        feed_dict = {}
        for name in batch_data.conf.user_slot_names + batch_data.conf.item_slot_names:
            ft = batch_data.tensor_dict[name]
            feed_dict[name] = create_tensor(ft.values, lod=ft.lod, place=place)
        pos = batch_data.pos().reshape([-1, 1])
        feed_dict['pos'] = create_tensor(pos, lod=batch_data.lod(), place=place)
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
        pos = batch_data.pos().reshape([-1, 1])
        feed_dict['pos'] = create_tensor(pos, lod=batch_data.lod(), place=place)
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
            elif args.model == 'BiRNN':
                model = BiRNN(conf, npz_config)
            elif args.model == 'Trans':
                model = Transformer(conf, npz_config, num_blocks=2, num_head=4)

            gpu_id = (0 if args.use_cuda == 1 else -1)
            if args.model in ['DNN', 'UniRNN']:
                algorithm = GenAlgorithm(model, optimizer=conf.optimizer, lr=conf.lr, gpu_id=gpu_id)
                td_ct = GenComputationTask(algorithm, model_dir=conf.model_dir, mode=args.train_mode, scope=scope)
            elif args.model in ['BiRNN', 'Trans']:
                algorithm = EvalAlgorithm(model, optimizer=conf.optimizer, lr=conf.lr, gpu_id=gpu_id)
                td_ct = EvalComputationTask(algorithm, model_dir=conf.model_dir, mode=args.train_mode, scope=scope)

    ### other tasks
    if args.task == 'test':
        test(td_ct, args, conf, None, td_ct.ckp_step)
        exit()
    elif args.task == 'eval':
        return td_ct

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
        fetch_dict = td_ct.train(SLFeedConvertor.train_test(batch_data))
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
        fetch_dict = td_ct.test(SLFeedConvertor.train_test(batch_data))
        auc_metric.add(labels=np.array(fetch_dict['click_id']).flatten(),
                        y_scores=np.array(fetch_dict['click_prob'])[:, 1])
        batch_id += 1

    add_scalar_summary(summary_writer, epoch_id, 'test/auc', auc_metric.overall_auc())


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args) 


