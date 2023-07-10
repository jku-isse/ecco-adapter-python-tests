from matplotlib import pyplot as plt
import evaluation_constants

def plot_warmup(step_width, results_file, averaged_episode_rewards, averaged_episode_steps):
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

def plot(period_length, step_width, results_file, averaged_episode_rewards, averaged_episode_steps):
    # num episodes
    p0 = period_length[0]
    p1 = int(period_length[1]/step_width)
    p2 = int(period_length[2]/step_width)
    p3 = int(period_length[3]/step_width)
    phase1 = range(p0, p1 + 1)
    phase2 = range(p1, p2 + 1)
    phase3 = range(p2, p3 + 1)
    total = range(len(averaged_episode_steps))
    fig, ax = plt.subplots(2)
    ax[0].plot(total, averaged_episode_rewards, label="Actual value")
    ax[0].set(xlabel='Episodes (in {})'.format(step_width),
              ylabel='Avg. reward',
              title='Reward')
    ax[0].plot(phase1, len(phase1) * [evaluation_constants.PHASE1_REWARD], 'g--', label="RandomAgent performance")
    ax[0].plot(phase2, len(phase2) * [evaluation_constants.PHASE2_REWARD], 'g--')
    ax[0].plot(phase3, len(phase3) * [evaluation_constants.PHASE3_REWARD], 'g--')
    # color phases area
    ax[0].axvspan(p0, p1, alpha=0.2, color='green')
    ax[0].axvspan(p1, p2, alpha=0.2, color='orange')
    ax[0].axvspan(p2, p3, alpha=0.2, color='red')
    ax[0].grid()
    ax[0].legend(bbox_to_anchor=(-0.03, 1.02, 1, 0.2), loc="lower left")

    ax[1].plot(total, averaged_episode_steps)
    ax[1].set(xlabel='Episodes (in {})'.format(step_width),
              ylabel='Avg. steps',
              title='Steps')
    ax[1].plot(phase1, len(phase1) * [evaluation_constants.PHASE1_STEPS], 'g--')
    ax[1].plot(phase2, len(phase2) * [evaluation_constants.PHASE2_STEPS], 'g--')
    ax[1].plot(phase3, len(phase3) * [evaluation_constants.PHASE3_STEPS], 'g--')
    # color phases area
    ax[1].axvspan(p0, p1, alpha=0.2, color='green')
    ax[1].axvspan(p1, p2, alpha=0.2, color='orange')
    ax[1].axvspan(p2, p3, alpha=0.2, color='red')
    ax[1].grid()



    plt.tight_layout()
    fig.savefig(results_file)
    plt.close(fig)


def plot_simple(step_width, results_file, averaged_episode_rewards, averaged_episode_steps):
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