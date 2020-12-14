import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np


REWMEAN = 0
DIFFICULTY = 1
WINDOWREW = 2

titles = ['Episode Rewards Mean', 
'Difficulty of Rule-Based Agent in 1v1',
'Mean Episode Reward over last 20 Episodes']
ylabels = ['Episode Reward Mean', 'Difficulty', 'Window Reward Mean']

colors = [
    '#5d42f5',
    '#f54242',
    '#2ecc71',
    '#f1c40f',
    '#42d1f5',
    '#e37e19',
    '#2c3e50',
    '#7f8c8d',
    '#ffd32a',
    '#ef5777',
    '#0be881',
    '#0fbcf9'
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

    path = sys.argv[1]
    with open(path, 'rb') as pickle_file:
        logs_dict = pickle.load(pickle_file)
        while True:
            try:
                if logs_dict['episode'] % 10 == 0:
                    print(pretty_print(logs_dict))
                ws = logs_dict['episode_window_size']
                timesteps.append(logs_dict['timesteps'])
                mean_ws_episode_rewards.append(sum(logs_dict['last_window_size_rewards']) / ws)
                eprewmeans.append(np.mean(logs_dict['episode_rewards']))
                difficulties.append(logs_dict['difficulty'])
                logs_dict = pickle.load(pickle_file)
            except EOFError:
                break
    ys = [eprewmeans, difficulties, mean_ws_episode_rewards]
    # for smoothing in [10, 12, 14, 16, 18, 20]:
    plt.plot(
        timesteps[smoothing:],
        [np.mean(ys[config][i-smoothing:i+1]) for i in range(len(timesteps)-smoothing)]
    )
    plt.title(titles[config]+' (smoothing=%d)'%smoothing)
    plt.xlabel('Timestep #')
    plt.ylabel(ylabels[config])
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

    plots = []
    for i in range(len(eval_rew_period_sums[0])):
        yaxis_data = eval_rew_period_sums[:,i]
        plot_i, = plt.plot(
            timesteps[added_smoothing:],
            [np.mean(yaxis_data[i-added_smoothing:i]) for i in range(len(timesteps)-added_smoothing)],
            color=colors[i]
        )
        plots.append(plot_i)

    plt.legend(
        labels=["{:.1f}".format(l) for l in np.linspace(0, 1, 10)],
        handles=plots
    )
    plt.title('Eval Results')
    plt.xlabel('Timestep #')
    plt.ylabel('Cumulative Reward Sum over 16 episodes')
    plt.show()


if __name__ == '__main__':
    # train_results(WINDOWREW)
    eval_results(sys.argv[1])


