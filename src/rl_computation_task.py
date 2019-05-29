"""
Defines the executors, programs, train entry, test entry and save/load models
"""
#!/usr/bin/env python
# coding=utf8
# File: train.py
from __future__ import print_function
import sys
import copy
import os
from os.path import exists, join
import numpy as np
from sklearn.metrics import roc_auc_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

PARL_DIR = os.environ.get('PARL_DIR', '')
assert exists(PARL_DIR), ('PARL_DIR', PARL_DIR, 'not exists')
sys.path.append(PARL_DIR)

import parl.layers as layers
from paddle import fluid
from paddle.fluid.framework import Variable

from utils import TracebackWrapper, save_pickle
from fluid_utils import executor_run_with_fetch_dict, parallel_executor_run_with_fetch_dict


class RLComputationTask(object):
    """
    For generation tasks
    """
    def __init__(self, algorithm, model_dir='', mode='single', mode_args=None, scope=None):
        """
        Args:
            algorithm: Algorithm object in PARL
        """
        self.alg = algorithm
        self.model_dir = model_dir
        self.mode = mode
        self.mode_args = mode_args
        self.ckp_step = -1
        self.use_cuda = True if self.alg.gpu_id >= 0 else False

        self.scope = scope
        with fluid.scope_guard(self.scope):
            self._define_program()
            self._define_executor(mode)

    def _define_program(self):
        """
        Use fluid.unique_name to make sure train 
        and test are using the same params 
        if the model is not base on PARL.
        """
        self.train_program = fluid.Program()
        self.startup_program = fluid.Program()
        self.test_program = fluid.Program()
        self.inference_program = fluid.Program()        # only consider single mode
        self.infer_init_program = fluid.Program()        # only consider single mode
        self.infer_onestep_program = fluid.Program()        # only consider single mode

        with fluid.program_guard(self.train_program, self.startup_program):
            with fluid.unique_name.guard():
                self.train_outputs = self.alg.train()

        with fluid.program_guard(self.test_program, fluid.Program()):
            with fluid.unique_name.guard():
                self.test_outputs = self.alg.test()

        with fluid.program_guard(self.inference_program, fluid.Program()):
            with fluid.unique_name.guard():
                self.inference_outputs = self.alg.inference()

        with fluid.program_guard(self.infer_init_program, fluid.Program()):
            with fluid.unique_name.guard():
                self.infer_init_outputs = self.alg.infer_init()

        with fluid.program_guard(self.infer_onestep_program, fluid.Program()):
            with fluid.unique_name.guard():
                self.infer_onestep_outputs = self.alg.infer_onestep()

    def _define_executor(self, mode):
        """
        define executors, run startup, and load saved models
        """
        if mode == 'single':
            place = fluid.CUDAPlace(0) if self.use_cuda else fluid.CPUPlace()
            self.base_exe = fluid.Executor(place)
            self.base_exe.run(self.startup_program)
            self.ckp_step = self.load_model(self.model_dir)
            self.train_exe = self.base_exe
            self.test_exe = self.base_exe

        elif mode == 'parallel':
            place = fluid.CUDAPlace(0) if self.use_cuda else fluid.CPUPlace()
            self.base_exe = fluid.Executor(place)
            self.base_exe.run(self.startup_program)
            self.ckp_step = self.load_model(self.model_dir)
            self._define_parallel_executor(self.train_program, self.test_program)

        elif mode == 'pserver':
            self._define_pserver_executor(self.mode_args, self.train_program, self.startup_program, self.test_program)

    def _define_parallel_executor(self, train_program, test_program):
        strategy = fluid.ExecutionStrategy()
        if self.use_cuda:
            strategy.num_threads = 1            # otherwise it will crash in GPU mode. 
        # strategy.allow_op_delay = False
        build_strategy = fluid.BuildStrategy()
        # build_strategy.reduce_strategy = fluid.BuildStrategy().ReduceStrategy.AllReduce
        # build_strategy.reduce_strategy = fluid.BuildStrategy().ReduceStrategy.Reduce    # runs faster
        loss = self.train_outputs['fetch_dict']['loss']
        self.train_exe = fluid.ParallelExecutor(use_cuda=self.use_cuda, 
                                                loss_name=loss.name,
                                                main_program=train_program,
                                                exec_strategy=strategy,
                                                build_strategy=build_strategy,
                                                scope=self.scope)
        self.test_exe = fluid.ParallelExecutor(use_cuda=self.use_cuda, 
                                                share_vars_from=self.train_exe,
                                                main_program=test_program,
                                                exec_strategy=strategy,
                                                build_strategy=build_strategy,
                                                scope=self.scope)

    def _define_pserver_executor(self, pserver_args, train_program, startup_program, test_program):  
        sync_mode = pserver_args['sync_mode']
        role = pserver_args['role']
        trainer_id = pserver_args['trainer_id'] # get actual trainer id here
        trainers = pserver_args['trainers']
        current_endpoint = pserver_args['current_endpoint']
        endpoints = pserver_args['endpoints']       # ip:port,ip:port or ip,ip;port

        def _process_endpoints(endpoints):
            if not ';' in endpoints:
                return endpoints
            ips, port = endpoints.split(';')
            return ','.join(['%s:%s' % (ip, port) for ip in ips.split(',')])
        endpoints = _process_endpoints(endpoints)

        dist_config = fluid.DistributeTranspilerConfig()
        dist_config.slice_var_up = False
        t = fluid.DistributeTranspiler(config=dist_config)
        t.transpile(trainer_id, 
                    program=train_program, 
                    startup_program=startup_program,
                    pservers=endpoints, 
                    trainers=trainers, 
                    sync_mode=sync_mode)

        if role == "PSERVER":
            ps_prog, ps_startup = t.get_pserver_programs(current_endpoint)
            # use CPU to execute pserver
            self.base_exe = fluid.Executor(fluid.CPUPlace())
            self.base_exe.run(ps_startup)
            self.ckp_step = self.load_model(self.model_dir)     # PSERVER to load model, and TRAINER to save model
            self.base_exe.run(ps_prog)   # will hold on here
            exit()

        elif role == "TRAINER":
            place = fluid.CUDAPlace(0) if self.use_cuda else fluid.CPUPlace()
            self.base_exe = fluid.Executor(place)
            self.base_exe.run(startup_program)
            train_program = t.get_trainer_program()

            self.ckp_step = self.get_lastest_checkpoint(self.model_dir)

            self._define_parallel_executor(train_program, test_program)

    ###################
    ### main functions
    ###################

    def train(self, list_feed_dict):
        """train"""
        with fluid.scope_guard(self.scope):
            self.alg.before_every_batch()
        if self.mode == 'single':
            assert len(list_feed_dict) == 1
            fetch_dict = executor_run_with_fetch_dict(self.train_exe, 
                                                program=self.train_program,
                                                fetch_dict=self.train_outputs['fetch_dict'],
                                                feed=list_feed_dict[0],
                                                return_numpy=False,
                                                scope=self.scope)

        elif self.mode == 'parallel' or self.mode == 'pserver':
            fetch_dict = parallel_executor_run_with_fetch_dict(self.train_exe,
                                                         fetch_dict=self.train_outputs['fetch_dict'],
                                                         feed=list_feed_dict,
                                                         return_numpy=False)
        with fluid.scope_guard(self.scope):
            self.alg.after_every_batch()
        return fetch_dict

    def test(self, list_feed_dict):
        """test"""
        if self.mode == 'single':
            assert len(list_feed_dict) == 1
            return executor_run_with_fetch_dict(self.test_exe, 
                                                program=self.test_program,
                                                fetch_dict=self.test_outputs['fetch_dict'],
                                                feed=list_feed_dict[0],
                                                return_numpy=False,
                                                scope=self.scope)

        elif self.mode == 'parallel' or self.mode == 'pserver':
            return parallel_executor_run_with_fetch_dict(self.test_exe,
                                                         fetch_dict=self.test_outputs['fetch_dict'],
                                                         feed=list_feed_dict,
                                                         return_numpy=False)

    def inference(self, feed_dict):
        """inference"""
        return executor_run_with_fetch_dict(self.base_exe, 
                                            program=self.inference_program,
                                            fetch_dict=self.inference_outputs['fetch_dict'],
                                            feed=feed_dict,
                                            return_numpy=False,
                                            scope=self.scope)

    def infer_init(self, feed_dict):
        """inference only the init part"""
        return executor_run_with_fetch_dict(self.base_exe, 
                                            program=self.infer_init_program,
                                            fetch_dict=self.infer_init_outputs['fetch_dict'],
                                            feed=feed_dict,
                                            return_numpy=False,
                                            scope=self.scope)

    def infer_onestep(self, feed_dict):
        """inference the gru-unit by one step"""
        return executor_run_with_fetch_dict(self.base_exe, 
                                            program=self.infer_onestep_program,
                                            fetch_dict=self.infer_onestep_outputs['fetch_dict'],
                                            feed=feed_dict,
                                            return_numpy=False,
                                            scope=self.scope)

    ##############
    ### utils
    ##############

    def run_startup(self):
        """will set params to random values"""
        with fluid.scope_guard(self.scope):
            self.base_exe.run(self.startup_program)
        logging.info('run startup')

    def save_infer_init_model(self, save_dirname):
        """save infer_init model"""
        feed_names = self.infer_init_outputs['feed_names']
        fetch_names = self.infer_init_outputs['fetch_dict'].keys()
        fluid.io.save_inference_model(save_dirname, 
                                      feed_names, 
                                      self.infer_init_outputs['fetch_dict'].values(), 
                                      self.base_exe, 
                                      main_program=self.infer_init_program)
        print ('saved inference model to', save_dirname)
        return feed_names, fetch_names

    def save_infer_onstep_model(self, save_dirname):
        """save infer_onestep model"""
        feed_names = self.infer_onestep_outputs['feed_names']
        fetch_names = self.infer_onestep_outputs['fetch_dict'].keys()
        fluid.io.save_inference_model(save_dirname, 
                                      feed_names, 
                                      self.infer_onestep_outputs['fetch_dict'].values(), 
                                      self.base_exe, 
                                      main_program=self.infer_onestep_program)
        print ('saved inference model to', save_dirname)
        return feed_names, fetch_names

    def save_model(self, path, checkpoint_step):
        """save network model"""
        if not exists(path):
            os.makedirs(path)
        with fluid.scope_guard(self.scope):
            fluid.io.save_params(executor=self.base_exe,
                                 dirname=path,
                                 main_program=self.train_program,
                                 filename='model-%d.ckp' % checkpoint_step)
        logging.info('==> Model saved to %s' % path)

    def load_model(self, path, ckp_step=None):
        """
        return -1 if not found
        """
        def _load_model(path, ckp_step):
            file = join(path, 'model-%d.ckp' % ckp_step)
            assert exists(file), file
            with fluid.scope_guard(self.scope):
                fluid.io.load_params(executor=self.base_exe,
                                     dirname=path,
                                     main_program=self.train_program,
                                     filename='model-%d.ckp' % ckp_step)
            logging.info('==> Model loaded from %s (step = %d)' % (path, ckp_step))

        if ckp_step is None:
            ckp_step = self.get_lastest_checkpoint(path)
        if ckp_step is None:
            logging.info('==> Model loaded from %s (not found, skipped)' % path)
            return -1
        else:
            _load_model(path, ckp_step)
            return ckp_step
    
    def get_lastest_checkpoint(self, path):
        """
        search lastest checkpoint with model-*.ckp under the give path
        return None if not found
        """
        if not exists(path):
            return None

        files = os.listdir(path)
        prefix = 'model-'
        suffix = '.ckp'
        last_ckp_step = None
        for f in files:
            if not (f.startswith(prefix) and f.endswith(suffix)):
                continue
            ckp_step = f[len(prefix):-len(suffix)]
            if not ckp_step.isdigit():
                continue
            ckp_step = int(ckp_step)
            if last_ckp_step is None:
                last_ckp_step = ckp_step
            else:
                last_ckp_step = max(last_ckp_step, ckp_step)
        return last_ckp_step







