import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np


REWMEAN = 0
DIFFICULTY = 1
WINDOWREW = 2

titles = ['Moving Episode Reward Mean over Last 100 Episodes', 
'Difficulty of Rule-Based Agent in 1v1',
'Sum of Last [Window Size] Rewards']
ylabels = ['Episode Reward Mean', 'Difficulty', 'Window Reward Sum']

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
def pretty_print(timesteps, eprewmean_buf, rewlen, awsr, difficulty):
  print('timesteps', timesteps)
  print('ep reward mean (last 100 episodes)', eprewmean_buf)
  print('length of rewards arr', rewlen)
  print('sum of last (window_size=1e4,1e5) rewards', awsr)
  print('difficulty', difficulty)
  print('===')


def train_results(config, smoothing=0):
    # use pickle path as first argument
    timesteps = []
    eprewmeans = []
    sum_of_last_windowsize_rewards=[]
    difficulties = []

    path = sys.argv[1]
    with open(path, 'rb') as pickle_file:
        logs_list = pickle.load(pickle_file)
        while True:
            try:
                timesteps.append(logs_list[0])
                eprewmeans.append(logs_list[1])
                sum_of_last_windowsize_rewards.append(logs_list[3])
                #if logs_list[3] > 0:
                print(pretty_print(*logs_list))
                difficulties.append(logs_list[4])
                logs_list = pickle.load(pickle_file)
            except EOFError:
                break
    ys = [eprewmeans, difficulties, sum_of_last_windowsize_rewards]
    for smoothing in [10, 12, 14, 16, 18, 20]:
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
    train_results(WINDOWREW)
    #eval_results(sys.argv[1])


