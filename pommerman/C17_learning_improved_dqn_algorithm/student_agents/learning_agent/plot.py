from matplotlib import pyplot as plt
import evaluation_constants

def plot_warmup(step_width, results_file, averaged_episode_rewards, averaged_episode_steps, eps_history):
    t = range(len(averaged_episode_steps))
    fig, ax = plt.subplots(2)
    ax[0].plot(t, averaged_episode_rewards, label="Actual value")
    ax[0].set(xlabel='Episodes (in {})'.format(step_width),
              ylabel='Avg. reward',
              title='Reward')
    # baseline reward
    ax[0].plot(t, len(t) * [evaluation_constants.WARMUP_REWARD], 'g--', label="RandomAgent performance")
    # p2 = ax[0].plot(t, len(t)*[evaluation_constants.WARMUP_STOP_REWARD], 'r--', label="Threshold value")
    ax[0].grid()
    ax[0].legend(bbox_to_anchor=(-0.03, 1.02, 1, 0.2), loc="lower left")
    ax[1].plot(t, averaged_episode_steps)
    ax[1].set(xlabel='Episodes (in {})'.format(step_width),
              ylabel='Avg. steps',
              title='Steps')
    # baseline steps
    ax[1].plot(t, len(t)*[evaluation_constants.WARMUP_STEPS], 'g--')
    # ax[1].plot(t, len(t) * [evaluation_constants.WARMUP_STOP_STEPS], 'r--')
    ax[1].grid()


    plt.tight_layout()
    fig.savefig(results_file)
    plt.close(fig)

def plot(step_width, results_file, averaged_episode_rewards, averaged_episode_steps, eps_history):
    t = range(len(averaged_episode_steps))
    fig, ax = plt.subplots(3)
    ax[0].plot(t, averaged_episode_rewards)
    ax[0].set(xlabel='Episodes (in {})'.format(step_width),
              ylabel='Avg. reward',
              title='Reward')

    ax[0].grid()
    ax[1].plot(t, averaged_episode_steps)
    ax[1].set(xlabel='Episodes (in {})'.format(step_width),
              ylabel='Avg. steps',
              title='Steps')
    ax[1].grid()

    plt.tight_layout()
    fig.savefig(results_file)
    plt.close(fig)


def plot_simple(step_width, results_file, averaged_episode_rewards, averaged_episode_steps, eps_history):
    t = range(len(averaged_episode_steps))
    fig, ax = plt.subplots(2)
    ax[0].plot(t, averaged_episode_rewards, label="Actual value")
    ax[0].set(xlabel='Episodes (in {})'.format(step_width),
              ylabel='Avg. reward',
              title='Reward')
    # baseline reward
    p1 = ax[0].plot(t, len(t) * [evaluation_constants.SIMPLE_REWARD], 'g--', label="RandomAgent performance")
    p2 = ax[0].plot(t, len(t)*[evaluation_constants.SIMPLE_STEPS], 'r--', label="Threshold value")
    ax[0].grid()
    ax[0].legend(bbox_to_anchor=(-0.03, 1.02, 1, 0.2), loc="lower left")
    ax[1].plot(t, averaged_episode_steps)
    ax[1].set(xlabel='Episodes (in {})'.format(step_width),
              ylabel='Avg. steps',
              title='Steps')
    # baseline steps
    ax[1].plot(t, len(t)*[evaluation_constants.WARMUP_STEPS], 'g--')
    ax[1].plot(t, len(t) * [evaluation_constants.WARMUP_STOP_STEPS], 'r--')
    ax[1].grid()


    plt.tight_layout()
    fig.savefig(results_file)
    plt.close(fig)