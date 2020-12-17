import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np


REWMEAN = 0
DIFFICULTY = 1
WINDOWREW = 2

titles = ['Training Rewards', 
'Difficulty of Rule-Based Agent in 1v1',
'Mean Episode Reward over last 20 Episodes']
ylabels = ['Episode Mean Reward', 'Difficulty', 'Window Reward Mean']

a, b = 0.0, 0.0

colors = [
    '#0be881',   
    '#2c3e50',  
    '#2ecc71',  
    '#42d1f5',  
    '#5d42f5',   
    '#7f8c8d',  
    '#ffd32a',
    '#f54242',    
    '#e37e19', 
    '#ef5777',  
]

# timesteps
# ep reward mean
# ep len mean
# difficulty
def pretty_print(d):
    for k in d:
        print(k, d[k])

def train_results(config, smoothing=0):
    # use pickle path as first argument
    timesteps = []
    eprewmeans = []
    mean_ws_episode_rewards=[] # mean rewards per episode over last [window size] episodes
    difficulties = []

    ws = 0
    global a,b

    path = sys.argv[1]
    with open(path, 'rb') as pickle_file:
        logs_dict = pickle.load(pickle_file)
        while True:
            try:
                if logs_dict['episode'] % 10 == 0:
                    print(pretty_print(logs_dict))
                a, b = logs_dict['a'], logs_dict['b']
                ws = logs_dict['episode_window_size']
                timesteps.append(logs_dict['timesteps'])
                # mean_ws_episode_rewards.append(sum(logs_dict['last_window_size_rewards']) / ws)
                eprewmeans.append(np.mean(logs_dict['episode_rewards']))
                difficulties.append(logs_dict['difficulty'])
                logs_dict = pickle.load(pickle_file)
            except EOFError:
                break
    ys = [eprewmeans, difficulties, mean_ws_episode_rewards]
    # for smoothing in [10, 12, 14, 16, 18, 20]:
    plt.plot(
        timesteps[smoothing:],
        [np.mean(ys[config][i-smoothing:i+1]) for i in range(smoothing, len(timesteps))]
    )
    smooth_str = '(smoothing=%d)'%smoothing if smoothing > 2 else ''
    plt.title(titles[config]+' a=%.1f,b=%.1f '%(a,b) + smooth_str)
    plt.xlabel('Timestep #')
    plt.ylabel(ylabels[config])
    plt.show()

def prob_distr():
    global a,b
    path = sys.argv[1]
    distr = []
    with open(path, 'rb') as pickle_file:
        logs_dict = pickle.load(pickle_file)
        while True:
            try:
                a, b = logs_dict['a'], logs_dict['b']
                distr = logs_dict['probabilities']
                logs_dict = pickle.load(pickle_file)
            except EOFError:
                break
    # for smoothing in [10, 12, 14, 16, 18, 20]:
    plt.bar(
        x=np.linspace(0.1, 0.9, 10),
        height=distr,width=0.1
    )
    plt.title('Final Difficulty Distribution for a=%.1f,b=%.1f '%(a,b))
    plt.xlabel('Difficulty')
    plt.ylabel('Probability')
    plt.show()


def eval_results(path, added_smoothing=3):
    timesteps = []
    eval_rew_period_sums = []

    with open(path, 'rb') as pickle_file:
        logs_list = pickle.load(pickle_file)
        while True:
            try:
                timesteps.append(logs_list[0])
                eval_rew_period_sums.append(logs_list[2])
                logs_list = pickle.load(pickle_file)
            except EOFError:
                break
    eval_rew_period_sums = np.array(eval_rew_period_sums)
    print(eval_rew_period_sums.shape)

    plots = []
    for i in range(len(eval_rew_period_sums[0])):
        yaxis_data = eval_rew_period_sums[:,i]
        plot_i, = plt.plot(
            timesteps[added_smoothing:],
            [np.mean(yaxis_data[i-added_smoothing:i]) for i in range(added_smoothing, len(timesteps))],
            color=colors[i]
        )
        plots.append(plot_i)

    plt.legend(
        labels=["{:.1f}".format(l) for l in np.linspace(0.1, 0.9, 10)],
        handles=plots
    )
    print(a,b)
    smooth_str = '(smoothing=%d)' % added_smoothing if added_smoothing > 2 else ''
    plt.title('Eval Results' + ' a=%.1f,b=%.1f '%(a,b) + smooth_str)
    plt.xlabel('Timestep #')
    plt.ylabel('Cumulative Reward Sum over 16 episodes')
    plt.show()


k = 5 # last k episodes to smooth over
rdi = 20 # reward difference interval, in episodes

def calc_deltas(rews):
    rewdiffs = []
    for i, diffrew in enumerate(rews):
        p = [np.mean(diffrew[j-rdi-k:j-rdi]) for j in range(rdi+k, len(diffrew))]
        l = [np.mean(diffrew[j-k:j]) for j in range(rdi+k, len(diffrew))]
        rewdiffs.append(np.array(l) - np.array(p))
    return rewdiffs

# the amount of positive reward gain over 20 episodes, by difficulty, over time.
def delta_spec(smoothing=1):
    # use pickle path as first argument
    timesteps = [[] for i in range(10)]
    rews = [[] for i in range(10)]

    path = sys.argv[1]
    with open(path, 'rb') as pickle_file:
        logs_dict = pickle.load(pickle_file)
        while True:
            try:
                d = int(10 * logs_dict['difficulty'] + 0.5)
                timesteps[d].append(logs_dict['timesteps'])
                rews[d].append(np.mean(logs_dict['episode_rewards']))
                logs_dict = pickle.load(pickle_file)
            except EOFError:
                break
    rews = np.array(rews)
    rewdiffs = calc_deltas(rews)
    plots = []
    for i in range(10):
        yaxis_data = rewdiffs[i]
        ti = timesteps[i][rdi+k:]
        plot_i, = plt.plot(
            ti[smoothing:],
            [np.mean(yaxis_data[j-smoothing:j]) for j in range(smoothing, len(ti))],
            color=colors[i]
        )
        plots.append(plot_i)

    plt.legend(
        labels=["{:.1f}".format(l) for l in np.linspace(0.1, 0.9, 10)],
        handles=plots
    )
    plt.title('Smoothed Reward Difference along Sliding Window=%d Episodes (smoothing=%d)' % (rdi, smoothing))
    plt.xlabel('Timestep #')
    plt.ylabel('Moving Difference in Reward from last %d episodes' % rdi)
    plt.show()


# gaussian reward obtained for each separate difficulty
def gaussian_eval(smoothing=10):
        # use pickle path as first argument
    timesteps = [[] for i in range(10)]
    rews = [[] for i in range(10)]

    path = sys.argv[1]
    with open(path, 'rb') as pickle_file:
        logs_dict = pickle.load(pickle_file)
        while True:
            try:
                a, b = logs_dict['a'], logs_dict['b']
                d = int(10 * logs_dict['difficulty'] + 0.5)
                timesteps[d].append(logs_dict['timesteps'])
                rews[d].append(np.mean(logs_dict['episode_rewards']))
                logs_dict = pickle.load(pickle_file)
            except EOFError:
                break
    rews = np.array(rews)
    plots = []
    for i in range(10):
        yaxis_data = rews[i]
        ti = timesteps[i]
        plot_i, = plt.plot(
            ti[smoothing:],
            [np.mean(yaxis_data[j-smoothing:j]) for j in range(smoothing, len(ti))],
            color=colors[i]
        )
        plots.append(plot_i)

    plt.legend(
        labels=["{:.1f}".format(l) for l in np.linspace(0.1, 0.9, 10)],
        handles=plots
    )
    plt.title('Curriculum Train Results (smoothing=%d)' % smoothing)
    plt.xlabel('Timestep #')
    plt.ylabel('Average Episode Reward')
    plt.show()


if __name__ == '__main__':
        prob_distr()
        train_results(REWMEAN, smoothing=1)
        eval_results(sys.argv[2], added_smoothing=1)
        # gaussian_eval(5)
        # delta_spec(3)

    


