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
from utils import (sequence_unconcat, BatchData, add_scalar_summary, PatternCounter)

import _init_paths

from src.gen_net import DNN, UniRNN
from src.gen_algorithm import GenAlgorithm
from src.gen_computation_task import GenComputationTask

from src.utils import (read_json, print_args, tik, tok, save_pickle, threaded_generator, 
                        AUCMetrics, AssertEqual, SequenceRMSEMetrics, SequenceCorrelationMetrics)
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
                        choices = ['train', 'test', 'eps_greedy_sampling', 'evaluate'],
                        type = str, 
                        help = "")
    
    # model settings
    parser.add_argument('--model', type=str, choices=['DNN', 'UniRNN'], help='')
    parser.add_argument('--eval_exp', type=str, help='')
    parser.add_argument('--eval_model', type=str, help='')
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
    def eps_greedy_sampling(batch_data, eps):
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

    @staticmethod
    def softmax_sampling(batch_data, eta):
        place = fluid.CPUPlace()
        feed_dict = {}
        for name in batch_data.conf.user_slot_names + batch_data.conf.item_slot_names:
            ft = batch_data.tensor_dict[name]
            feed_dict[name] = create_tensor(ft.values, lod=ft.lod, place=place)

        decode_len = batch_data.decode_len().reshape([-1, 1]).astype('int64')
        lod = [seq_len_2_lod([1] * len(decode_len))]
        feed_dict['decode_len'] = create_tensor(decode_len, lod=lod, place=place)
        eta = np.array([eta]).astype('float32')
        feed_dict['eta'] = create_tensor(eta, lod=[], place=place)
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
            if args.model == 'DNN':
                model = DNN(conf, npz_config)
            elif args.model == 'UniRNN':
                model = UniRNN(conf, npz_config)

            algorithm = GenAlgorithm(model, optimizer=conf.optimizer, lr=conf.lr, gpu_id=(0 if args.use_cuda == 1 else -1))
            td_ct = GenComputationTask(algorithm, model_dir=conf.model_dir, mode=args.train_mode, scope=scope)


    # get eval model
    eval_args = copy.deepcopy(args)
    eval_args.exp = args.eval_exp
    eval_args.model = args.eval_model
    eval_args.task = 'eval'
    eval_td_ct = eval_entry_func(eval_args)

    ### other tasks
    if args.task == 'test':
        test(td_ct, args, conf, None, td_ct.ckp_step)
        exit()
    elif args.task == 'eps_greedy_sampling':
        eps_greedy_sampling(td_ct, eval_td_ct, args, conf, None, td_ct.ckp_step)
        exit()
    elif args.task == 'evaluate':
        evaluate(td_ct, eval_td_ct, args, conf, td_ct.ckp_step)
        exit()

    ### start training
    summary_writer = tf.summary.FileWriter(conf.summary_dir)
    for epoch_id in range(td_ct.ckp_step + 1, conf.max_train_steps):
        train(td_ct, args, conf, summary_writer, epoch_id)
        td_ct.save_model(conf.model_dir, epoch_id)
        test(td_ct, args, conf, summary_writer, epoch_id)
        eps_greedy_sampling(td_ct, eval_td_ct, args, conf, summary_writer, epoch_id)


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
    seq_rmse_metric = SequenceRMSEMetrics()
    seq_correlation_metric = SequenceCorrelationMetrics()
    batch_id = 0
    for tensor_dict in data_gen:
        batch_data = BatchData(conf, tensor_dict)
        fetch_dict = td_ct.test(GenSLFeedConvertor.train_test(batch_data))
        click_id = np.array(fetch_dict['click_id']).flatten()
        click_prob = np.array(fetch_dict['click_prob'])[:, 1]
        click_id_unconcat = sequence_unconcat(click_id, batch_data.seq_lens())
        click_prob_unconcat = sequence_unconcat(click_prob, batch_data.seq_lens())
        auc_metric.add(labels=click_id, y_scores=click_prob)
        for sub_click_id, sub_click_prob in zip(click_id_unconcat, click_prob_unconcat):
            seq_rmse_metric.add(labels=sub_click_id, preds=sub_click_prob)
            seq_correlation_metric.add(labels=sub_click_id, preds=sub_click_prob)
        batch_id += 1

    add_scalar_summary(summary_writer, epoch_id, 'test/auc', auc_metric.overall_auc())
    add_scalar_summary(summary_writer, epoch_id, 'test/seq_rmse', seq_rmse_metric.overall_rmse())
    add_scalar_summary(summary_writer, epoch_id, 'test/seq_correlation', seq_correlation_metric.overall_correlation())


def eps_greedy_sampling(td_ct, eval_td_ct, args, conf, summary_writer, epoch_id):
    """eps_greedy_sampling"""
    dataset = NpzDataset(conf.test_npz_list, conf.npz_config_path, conf.requested_npz_names, if_random_shuffle=False)
    data_gen = dataset.get_data_generator(conf.batch_size)

    list_reward = []
    last_batch_data = BatchData(conf, data_gen.next())
    batch_id = 0
    for tensor_dict in data_gen:
        ### eps_greedy_sampling
        batch_data = BatchData(conf, tensor_dict)
        batch_data.set_decode_len(batch_data.seq_lens())
        batch_data.expand_candidates(last_batch_data, batch_data.seq_lens())

        fetch_dict = td_ct.eps_greedy_sampling(GenSLFeedConvertor.eps_greedy_sampling(batch_data, eps=0))
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

    add_scalar_summary(summary_writer, epoch_id, 'eps_greedy_sampling/reward-%s' % args.eval_exp, np.mean(list_reward))


def evaluate(td_ct, eval_td_ct, args, conf, epoch_id):
    """softmax_sampling"""
    np.random.seed(0)   # IMPORTANT. To have the same candidates, since the candidates is selected by np.random.choice.
    dataset = NpzDataset(conf.test_npz_list, conf.npz_config_path, conf.requested_npz_names, if_random_shuffle=False)
    batch_size = 250
    data_gen = dataset.get_data_generator(batch_size)

    max_batch_id = 200
    list_n = [1, 20, 40]
    dict_reward = {'eps_greedy':[], 'softmax':{n:[] for n in list_n}}
    p_counter = PatternCounter()
    last_batch_data = BatchData(conf, data_gen.next())
    for batch_id in range(max_batch_id):
        def get_list_wise_reward(batch_data, order):
            reordered_batch_data = batch_data.get_reordered(order)
            fetch_dict = eval_td_ct.inference(EvalFeedConvertor.inference(reordered_batch_data))
            reward = np.array(fetch_dict['click_prob'])[:, 1]
            reward_unconcat = sequence_unconcat(reward, [len(od) for od in order])
            return [np.sum(rw) for rw in reward_unconcat]

        def greedy_sampling(batch_data):
            fetch_dict = td_ct.eps_greedy_sampling(GenSLFeedConvertor.eps_greedy_sampling(batch_data, eps=0))
            sampled_id = np.array(fetch_dict['sampled_id']).reshape([-1])
            order = sequence_unconcat(sampled_id, batch_data.decode_len())
            list_wise_reward = get_list_wise_reward(batch_data, order)      # (b,)
            return order, list_wise_reward

        def softmax_sampling(batch_data, max_sampling_time):
            mat_list_wise_reward = []
            mat_order = []
            for i in range(max_sampling_time):
                fetch_dict = td_ct.softmax_sampling(GenSLFeedConvertor.softmax_sampling(batch_data, eta=0.1))
                sampled_id = np.array(fetch_dict['sampled_id']).reshape([-1])
                order = sequence_unconcat(sampled_id, batch_data.decode_len())
                list_wise_reward = get_list_wise_reward(batch_data, order)
                mat_order.append(order) 
                mat_list_wise_reward.append(list_wise_reward)
            mat_list_wise_reward = np.array(mat_list_wise_reward)   # (max_sampling_time, b)
            return mat_order, mat_list_wise_reward      # (max_sampling_time, b, var_seq_len), 

        tensor_dict = data_gen.next()
        batch_data = BatchData(conf, tensor_dict)
        batch_data.set_decode_len(batch_data.seq_lens())
        batch_data.expand_candidates(last_batch_data, batch_data.seq_lens())
        p_counter.add_log_pattern(batch_data)

        ### eps_greedy_sampling
        order, list_wise_reward = greedy_sampling(batch_data)
        dict_reward['eps_greedy'] += list_wise_reward
        p_counter.add_sampled_pattern('eps_greedy', batch_data, order)

        ### softmax_sampling
        max_sampling_time = np.max(list_n)
        mat_order, mat_list_wise_reward = softmax_sampling(batch_data, max_sampling_time)
        for n in list_n:
            dict_reward['softmax'][n] += np.max(mat_list_wise_reward[:n], 0).tolist()
            max_indice = np.argmax(mat_list_wise_reward[:n], 0)     # (b,)
            max_order = [mat_order[max_id][b_id] for b_id, max_id in enumerate(max_indice)]
            p_counter.add_sampled_pattern('softmax_%d' % n, batch_data, max_order)

        ### log
        if batch_id % 10 == 0:
            logging.info('batch_id:%d eps_greedy %f' % (batch_id, np.mean(dict_reward['eps_greedy'])))
            for n in list_n:
                logging.info('batch_id:%d softmax_%d %f' % (batch_id, n, np.mean(dict_reward['softmax'][n])))
            p_counter.Print()

        last_batch_data = BatchData(conf, tensor_dict)

    ### log
    logging.info('final eps_greedy %f' % np.mean(dict_reward['eps_greedy']))
    for n in list_n:
        logging.info('final softmax_%d %f' % (n, np.mean(dict_reward['softmax'][n])))
    p_counter.Print()

    ### save
    pickle_file = 'tmp/%s-eval_%s.pkl' % (args.exp, args.eval_model)
    p_counter.save(pickle_file)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)


