
import os
import time
import numpy as np
import pickle
import os.path as osp
import datetime
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
from baselines.common.policies import build_policy
from baselines.bench import monitor
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import importlib
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from baselines.ppo2.runner import Runner
import gfootball.env as football_env
from gfootball.curriculum_learning import load_logs

def constfn(val):
    def f(_):
        return val
    return f

def learn(network, FLAGS, eval_env = None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=10, load_path=None, model_fn=None, update_fn=None, init_fn=None, mpi_rank_weight=1, comm=None, 
            episode_window_size=20, stop=True,
            scenario='gfootball.scenarios.1_vs_1_easy',
            curriculum=np.linspace(0, 0.9, 10), b=0.2,
            eval_period=20, eval_episodes=1,
            **network_kwargs):
    '''
    Learn policy using PPO algorithm (https://arxiv.org/abs/1707.06347)
    Parameters:
    ----------
    network:                          policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                                     specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                                     tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                                     neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                                    See common/models.py/lstm for more details on using recurrent nets in policies
    env: baselines.common.vec_env.VecEnv     environment. Needs to be vectorized for parallel environment simulation.
                                      The environments produced by gym.make can be wrapped using baselines.common.vec_env.DummyVecEnv class.
    nsteps: int                       number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                                      nenv is number of environment copies simulated in parallel)
    ent_coef: float                   policy entropy coefficient in the optimization objective
    lr: float or function             learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the
                                      training and 0 is the end of the training.
    vf_coef: float                    value function loss coefficient in the optimization objective
    max_grad_norm: float or None      gradient norm clipping coefficient
    gamma: float                      discounting factor
    lam: float                        advantage estimation discounting factor (lambda in the paper)
    log_interval: int                 number of timesteps between logging events
    nminibatches: int                 number of training minibatches per update. For recurrent policies,
                                      should be smaller or equal than number of environments run in parallel.
    noptepochs: int                   number of training epochs per update
    cliprange: float or function      clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
                                      and 0 is the end of the training
    save_interval: int                number of timesteps between saving events
    load_path: str                    path to load the model from
    **network_kwargs:                 keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                                      For instance, 'mlp' network architecture has arguments num_hidden and num_layers.
    '''

    set_global_seeds(seed)

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    
    basic_builder = importlib.import_module(scenario, package=None)
    def build_builder_with_difficulty(difficulty):
        def builder_with_difficulty(builder):
            basic_builder.build_scenario(builder)
            builder.config().right_team_difficulty = difficulty
            builder.config().left_team_difficulty = difficulty
        return builder_with_difficulty
      

    def create_single_football_env(iprocess):
        """Creates gfootball environment."""
        env = football_env.create_environment(
            env_name=build_builder_with_difficulty(0), stacked=('stacked' in FLAGS.state),
            rewards=FLAGS.reward_experiment,
            logdir=logger.get_dir(),
            write_goal_dumps=FLAGS.dump_scores and (iprocess == 0),
            write_full_episode_dumps=FLAGS.dump_full_episodes and (iprocess == 0),
            render=FLAGS.render and (iprocess == 0),
            dump_frequency=50 if FLAGS.render and iprocess == 0 else 0)
        env = monitor.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(),
                                                                    str(iprocess)))
        return env

    env = SubprocVecEnv([
        (lambda _i=i: create_single_football_env(_i))
        for i in range(FLAGS.num_envs)
    ], context=None)
    
    policy = build_policy(env, network, **network_kwargs)

    average_window_size = episode_window_size * 16

    # Get the nb of env
    nenvs = FLAGS.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

    # Configure logger to log_ppo_timestamp formatted
    pickle_str = 'curriculum_ppo_' + '-'.join(str(datetime.datetime.now()).replace(':', ' ').split(' '))
    eval_pickle_str = pickle_str + '_eval'

    # open pickle file to append relevant data in binary
    pickle_dir = '/content/cs285_f2020_proj/football/pickled_data/'
    model_dir = '/content/cs285_f2020_proj/football/models/'

    # create dir for pickling & model save
    if not os.path.exists(pickle_dir): 
        os.makedirs(pickle_dir)
    if not os.path.exists(model_dir): 
        os.makedirs(model_dir)

    def make_file(file_path):
        if not os.path.exists(file_path):
            with open(file_path, 'w+'):
                print('made path', file_path)
    
    make_file(pickle_dir + pickle_str)
    # make_file(pickle_dir + eval_pickle_str)

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from baselines.ppo2.model import Model
        model_fn = Model

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, comm=comm, mpi_rank_weight=mpi_rank_weight)

    if load_path is not None:
        model.load(load_path)

    def create_single_football_env(iprocess, difficulty):
        """Creates gfootball environment."""
        env = football_env.create_environment(
            env_name=build_builder_with_difficulty(difficulty), stacked=('stacked' in FLAGS.state),
            rewards=FLAGS.reward_experiment,
            logdir=logger.get_dir(),
            write_goal_dumps=FLAGS.dump_scores and (iprocess == 0),
            write_full_episode_dumps=FLAGS.dump_full_episodes and (iprocess == 0),
            render=FLAGS.render and (iprocess == 0),
            dump_frequency=50 if FLAGS.render and iprocess == 0 else 0)
        env = monitor.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(),
                                                                    str(iprocess)))
        return env

    def make_runner(difficulty):
        vec_env = SubprocVecEnv([
            (lambda _i=i: create_single_football_env(_i, difficulty))
            for i in range(FLAGS.num_envs)
        ], context=None)
        print('vec env obs space', vec_env.observation_space)
        return env, Runner(env=vec_env, model=model, nsteps=nsteps, gamma=gamma, lam=lam) 

    # get next difficulty according to distribution outlined in probabilities.
    def get_next_difficulty():
        draw = np.random.choice(range(10), 1, p=curriculum_probabilities)
        return draw[0]

    # Instantiate the runner object
    # Curriculum difficulties start off as random.
    curriculum_probabilities = [0.1] * 10

    difficulty_idx = get_next_difficulty()
    env, runner = make_runner(curriculum[difficulty_idx])

    def make_eval_runner(difficulty):
        vec_env = SubprocVecEnv([
            (lambda _i=i: create_single_football_env(_i, difficulty))
            for i in range(FLAGS.num_envs, 2*FLAGS.num_envs)
        ], context=None)
        print('vec env obs space', vec_env.observation_space)
        return env, Runner(env=vec_env, model=model, nsteps=nsteps, gamma=gamma, lam=lam) 
    
    policy = build_policy(env, network, **network_kwargs)
     
    eprews = []
    rews_by_difficulty = [[] for i in range(10)]

    k = 5 # last k episodes to smooth over
    rdi = 20 # reward difference interval, in episodes

    def update_curriculum_probabilities():
            smart_mean = lambda l: np.mean(l) if l else 0
            prev_smoothed_rews = np.zeros(10)
            latest_smoothed_rews = np.zeros(10)
            for i, diffrewlist in enumerate(rews_by_difficulty):
                # mean the last k episode's rewards
                prev_smoothed_rews[i] = smart_mean(rews_by_difficulty[i][-rdi-k:-rdi])
                latest_smoothed_rews[i] = smart_mean(rews_by_difficulty[i][-k:])

            e_diff_rews = np.exp(b * (latest_smoothed_rews - prev_smoothed_rews))
            return e_diff_rews / np.sum(e_diff_rews)



    # eval_rews[i] will be all the rewards from evaluation i
    # eval_rews[i][j] will be rewards from evaluation i at difficulty j ~ 2:20 = (0.05, 0.95)
    eval_rews = [] 

    epinfobuf = deque(maxlen=100)

    if init_fn is not None:
        init_fn()

    # Start total timer
    tfirststart = time.perf_counter()

    # nupdates = total_timesteps//nbatch
    update = 0
    while True:
        update += 1
        assert nbatch % nminibatches == 0
        # Start timer
        tstart = time.perf_counter()
        # frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(0) # Constant LR, cliprange
        # Calculate the cliprange
        cliprangenow = cliprange(0)

        if update % log_interval == 0 and is_mpi_root: 
            logger.info('Stepping environment...')

        # Get minibatch
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
    
        if update % log_interval == 0 and is_mpi_root: 
            logger.info('Done.')

        rewards_this_episode = [i['r'] for i in epinfos]
        lengths_this_episode = [i['l'] for i in epinfos]

        print('episode rewards ep#', update, rewards_this_episode)
        eprews.extend(rewards_this_episode)
        epinfobuf.extend(epinfos)

        # for each minibatch calculate the loss and append it.
        mblossvals = []
        if states is None: # nonrecurrent version
            # Index of each element of batch_size
            # Create the indices array
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))


        # sum of last average_window_size rewards
        last_aws_rewards_sum = sum(eprews[-average_window_size:])
        rews_by_difficulty[difficulty_idx].append(np.sum(rewards_this_episode))

        # pickling
        pickle_data = {
          'episode' : update,
          'timesteps' : update*nsteps,
          'episode_rewards' : rewards_this_episode,
          'episode_window_size' : episode_window_size,
          'last_window_size_rewards' : eprews[-average_window_size:],
          'difficulty' : curriculum[difficulty_idx],
          'len_rewards_array' : len(eprews),
          'episode_lenths' : lengths_this_episode,
          'eval_period' : eval_period,
        }

        def dict_print(d):
          for k in d:
            print(k, d[k])

        dict_print(pickle_data)
        with open(pickle_dir + pickle_str, 'ab') as pickle_file:
            pickle.dump(pickle_data, pickle_file)                      

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.perf_counter()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))

        if update_fn is not None:
            update_fn(update)

        '''
        # every eval period run for eval_nsteps on every difficulty
        if update % eval_period == 1:
            # rews[i] = sum of rewards from eval_nsteps for difficulty index i
            eval_rews_period = [] # 2D array
            eval_rews_period_sum = [] # 1D array
            for difficulty_eval in curriculum[::2]:
                eval_env, eval_runner = make_eval_runner(difficulty_eval)
                eval_rewards_for_difficulty = []
                for k in range(eval_episodes):
                    # run nsteps for the number of eval episodes (nsteps * episodes)
                    eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, eval_epinfos = eval_runner.run() #pylint: disable=E0632
                    # append the array of all the rewards gotten for this difficulty in the episode.
                    eval_rewards_for_difficulty.extend([i['r'] for i in eval_epinfos])
                eval_rews_period.append(eval_rewards_for_difficulty)
                eval_rews_period_sum.append(sum(eval_rewards_for_difficulty))
                print("rews eval timstep", update*nsteps, "difficulty", difficulty_eval, eval_rewards_for_difficulty, "sum", eval_rews_period_sum[-1])

            eval_rews.append(eval_rews_period)
            eval_pickle_data = [
              update*nsteps, # timesteps for trainer
              eval_rews_period, # 2D array which contains all rewards gotten for all difficulties this eval period.
              eval_rews_period_sum
            ]
            with open(pickle_dir + eval_pickle_str, 'ab') as eval_pickle_file:
                pickle.dump(eval_pickle_data, eval_pickle_file)
            print('eval pickle dumped, u#', update)
        '''

        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("misc/serial_timesteps", update*nsteps)
            logger.logkv("misc/nupdates", update)
            logger.logkv("misc/total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("misc/explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('misc/time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv('loss/' + lossname, lossval)

            logger.dumpkvs()
        if save_interval and update % save_interval == 1:
            savepath = osp.join(model_dir, pickle_str)
            print('Saving to', savepath)
            model.save(savepath)

        curriculum_probabilities = update_curriculum_probabilities()
        print('new probability distr:', curriculum_probabilities)
        difficulty_idx = get_next_difficulty()
        print("NEXT DIFFICULTY:",curriculum[difficulty_idx])
        env, runner = make_runner(curriculum[difficulty_idx])
    return model
# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
