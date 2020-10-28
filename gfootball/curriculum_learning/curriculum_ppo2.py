
import os
import time
import numpy as np
import os.path as osp
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

def constfn(val):
    def f(_):
        return val
    return f

def learn(network, FLAGS, eval_env = None, seed=None, nsteps=2048, ent_coef=0.0, lr=3e-4,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, load_path=None, model_fn=None, update_fn=None, init_fn=None, mpi_rank_weight=1, comm=None, 
            average_window_size=int(1e5), stop=True,
            scenario='gfootball.scenarios.1_vs_1_easy',
            curriculum=np.linspace(0, 0.95, 20),
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

    # Get the nb of env
    nenvs = FLAGS.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from baselines.ppo2.model import Model
        model_fn = Model

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, comm=comm, mpi_rank_weight=mpi_rank_weight)

    if load_path is not None:
        model.load(load_path)

    def make_runner(difficulty):
        def create_single_football_env(iprocess):
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

        vec_env = SubprocVecEnv([
            (lambda _i=i: create_single_football_env(_i))
            for i in range(FLAGS.num_envs)
        ], context=None)
        print('vec env obs space', vec_env.observation_space)
        return env, Runner(env=vec_env, model=model, nsteps=nsteps, gamma=gamma, lam=lam) 
    # Instantiate the runner object
    env, runner = make_runner(curriculum[0])
    difficulty_idx = 0
    if eval_env is not None:
        eval_runner = Runner(env = eval_env, model = model, nsteps = nsteps, gamma = gamma, lam= lam)
 
    eprews = []
    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

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

        if update % log_interval == 0 and is_mpi_root: logger.info('Stepping environment...')

        # Get minibatch
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
        if eval_env is not None:
            eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, eval_epinfos = eval_runner.run() #pylint: disable=E0632
    
        if update % log_interval == 0 and is_mpi_root: logger.info('Done.')

        eprews.extend([i['r'] for i in epinfos])
        epinfobuf.extend(epinfos)
        if eval_env is not None:
            eval_epinfobuf.extend(eval_epinfos)

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

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.perf_counter()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))

        if update_fn is not None:
            update_fn(update)

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
            if eval_env is not None:
                logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]) )
                logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]) )
            logger.logkv('misc/time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv('loss/' + lossname, lossval)

            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and is_mpi_root:
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
        if difficulty_idx < len(curriculum)-1 and len(eprews) >= average_window_size and sum(eprews[-average_window_size:]) >= 0:
            difficulty_idx += 1
            print("\n\n\n\n\n=====================================\n",
                  "INCREASING DIFFICULTY TO",curriculum[difficulty_idx],
                  "\n===========================================\n\n\n\n\n\n")
            env, runner = make_runner(curriculum[difficulty_idx], model=model, nsteps=nsteps, gamma=gamma, lam=lam)
            if eval_env is not None:
                eval_runner = Runner(env = eval_env, model = model, nsteps = nsteps, gamma = gamma, lam= lam)
    return model
# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
