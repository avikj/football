# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs football_env on OpenAI's ppo2."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
from absl import app
from absl import flags
from baselines import logger
from baselines.bench import monitor
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from gfootball.curriculum_learning import curriculum_ppo2
from gfootball.curriculum_learning import curriculum_ab_ppo2
from gfootball.curriculum_learning import single_difficulty_ppo2
import gfootball.env as football_env
from gfootball.examples import models  


FLAGS = flags.FLAGS

flags.DEFINE_enum('state', 'extracted_stacked', ['extracted',
                                                 'extracted_stacked'],
                  'Observation to be used for training.')
flags.DEFINE_enum('reward_experiment', 'scoring,checkpoints',
                  ['scoring', 'scoring,checkpoints'],
                  'Reward to be used for training.')
flags.DEFINE_enum('policy', 'gfootball_impala_cnn', ['cnn', 'lstm', 'mlp', 'impala_cnn',
                                    'gfootball_impala_cnn'],
                  'Policy architecture')
flags.DEFINE_integer('num_timesteps', 500000,
                     'Number of timesteps to run for.')
flags.DEFINE_integer('num_envs', 8,
                     'Number of environments to run in parallel.')
flags.DEFINE_integer('nsteps', 1024, 'Number of environment steps per epoch; '
                     'batch size is nsteps * nenv')
flags.DEFINE_integer('noptepochs', 4, 'Number of updates per epoch.')
flags.DEFINE_integer('nminibatches', 8,
                     'Number of minibatches to split one epoch to.')
flags.DEFINE_integer('save_interval', 100,
                     'How frequently checkpoints are saved.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('lr', 0.00008, 'Learning rate') # og: 0.00008
flags.DEFINE_float('ent_coef', 0.01, 'Entropy coeficient')
flags.DEFINE_float('gamma', 0.993, 'Discount factor')
flags.DEFINE_float('cliprange', 0.27, 'Clip range')
flags.DEFINE_float('max_grad_norm', 0.5, 'Max gradient norm (clipping)')
flags.DEFINE_float('a', 0, 'the scalar on mean')
flags.DEFINE_float('b', 0, 'the scalar on variance')
flags.DEFINE_bool('render', False, 'If True, environment rendering is enabled.')
flags.DEFINE_bool('dump_full_episodes', False,
                  'If True, trace is dumped after every episode.')
flags.DEFINE_bool('dump_scores', False,
                  'If True, sampled traces after scoring are dumped.')
flags.DEFINE_string('load_path', None, 'Path to load initial checkpoint from.')
flags.DEFINE_string('scenario', 'gfootball.scenarios.1_vs_1_easy', 'Scenario to run')

def train(_):
  """Trains a PPO2 policy."""

  # Import tensorflow after we create environments. TF is not fork sake, and
  # we could be using TF as part of environment if one of the players is
  # controled by an already trained model.
  import tensorflow.compat.v1 as tf
  ncpu = multiprocessing.cpu_count()
  config = tf.ConfigProto(allow_soft_placement=True,
                          intra_op_parallelism_threads=ncpu,
                          inter_op_parallelism_threads=ncpu)
  config.gpu_options.allow_growth = True
  tf.Session(config=config).__enter__()

  print('Running scenario', FLAGS.scenario)
  print('a=%d,b=%d' % (FLAGS.a, FLAGS.b), 'timesteps=', FLAGS.num_timesteps)

  curriculum_ab_ppo2.learn(FLAGS.policy,
             FLAGS,
             seed=FLAGS.seed,
             nsteps=FLAGS.nsteps,
             nminibatches=FLAGS.nminibatches,
             noptepochs=FLAGS.noptepochs,
             max_grad_norm=FLAGS.max_grad_norm,
             gamma=FLAGS.gamma,
             ent_coef=FLAGS.ent_coef,
             lr=FLAGS.lr,
             a=FLAGS.a,
             b=FLAGS.b,
             num_timesteps=FLAGS.num_timesteps,
             log_interval=1,
             save_interval=FLAGS.save_interval,
             scenario=FLAGS.scenario,
             cliprange=FLAGS.cliprange,
             load_path=FLAGS.load_path)


if __name__ == '__main__':
  app.run(train)
