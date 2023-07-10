import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import math
import os
from torch.utils.tensorboard import SummaryWriter
import pickle
from pommerman.constants import Action

from learning_agent.net_input import featurize
from helper import create_random_env, parse_conf, path_to_instance_simple
from learning_agent.replay_memory import ReplayMemory, Transition
from reward_modifier import RewardModifier
from plot import plot, plot_warmup
from action_filter import ActionFilter
from jitter_filter import JitterFilter
from curriculum_learning import Curriculum

general_settings = None
algorithm_settings = None
store_mode = None
render_options = None
stats = None

memory = None
optimizer = None
policy_net = None
target_net = None
device = None

def select_action(state, n_actions, episode_count):
    if len(memory) > algorithm_settings['MIN_EXPERIENCE_REPLAY_SIZE'] and \
            algorithm_settings["EPS"] > algorithm_settings["EPS_END"]:
        if algorithm_settings["EPS_DEC_METHOD"] == "linear":
            algorithm_settings["EPS"] = algorithm_settings["EPS_START"] - episode_count* algorithm_settings["EPS_DECAY"]
        else:
            algorithm_settings['EPS'] = algorithm_settings['EPS_END'] + \
                                        (algorithm_settings['EPS_START'] - algorithm_settings['EPS_END']) * \
                math.exp(-1. * episode_count / algorithm_settings['EPS_DECAY'])
    sample = random.random()
    if sample > algorithm_settings['EPS']:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < algorithm_settings['MIN_EXPERIENCE_REPLAY_SIZE']:
        return torch.tensor([0])
    transitions = memory.sample(algorithm_settings['BATCH_SIZE'])
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    previous_states = torch.cat(batch.last_state)
    actions = torch.cat(batch.last_action)
    rewards = torch.cat(batch.reward)
    current_states = torch.cat(batch.current_state)
    terminal = torch.cat(batch.terminal)
    non_terminal = torch.tensor(tuple(map(lambda s: not s,
                                            batch.terminal)), device=device, dtype=torch.bool)

    state_action_values = policy_net(previous_states).gather(1, actions)

    # version without double Q-Learning
    agent_reward_per_action = target_net(current_states).max(1)[0].detach()

    # calculate the expected reward at current_states
    agents_expected_reward = torch.zeros(algorithm_settings['BATCH_SIZE'], device=device)
    agents_expected_reward[terminal] = rewards[terminal]
    agents_expected_reward[non_terminal] = rewards[non_terminal] + \
                                           algorithm_settings['GAMMA'] * agent_reward_per_action[non_terminal].squeeze()

    loss = F.mse_loss(state_action_values, agents_expected_reward.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    num = 0
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
        num += 1
    optimizer.step()

    return loss

def postprocess_action(action_filter, jitter_filter, last_action, env, agent_pos, trainee):
    real_action = last_action.item()
    possible_actions = list(range(env.action_space.n))
    possible_actions.remove(real_action)
    # agent cannot lay bomb
    if trainee.ammo == 0 and real_action == Action.Bomb.value:
        real_action = Action.Stop.value
        possible_actions.remove(real_action)
    if jitter_filter.has_jitter(x_pos=agent_pos[0], y_pos=agent_pos[1]):
        real_action = random.choice(possible_actions)
        possible_actions.remove(real_action)
    while not action_filter.isSafeAction(real_action) and len(possible_actions) > 0:
        real_action = random.choice(possible_actions)
        possible_actions.remove(real_action)
    return real_action


def train():
    global general_settings, algorithm_settings, store_mode, render_options, stats
    global memory, optimizer, policy_net, target_net, device
    os.environ['PYTHONHASHSEED'] = str(1)
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    general_settings, algorithm_settings, store_mode, render_options, stats = parse_conf()

    # use cuda if device is available, else cpu
    device = torch.device(general_settings['DEVICE'])
    print("Running on device: {}".format(device))

    # Define a tensorboard summary writer that writes to directory "results_path/tensorboard"
    writer = SummaryWriter(log_dir=os.path.join(stats["RESULTS_PATH"], 'tensorboard'))
    curriculum = Curriculum(algorithm_settings["PERIOD_LENGTH"])
    env, trainee, trainee_id, opponent, opponent_id = curriculum.get_env(1)
    last_state = env.reset()
    agent_pos = last_state[trainee_id]['position']
    last_state_trainee = last_state[trainee_id]
    # initialize modules needed
    action_filter = ActionFilter(last_state_trainee)
    jitter_filter = JitterFilter()
    reward_modifier = RewardModifier(last_state_trainee, curriculum.current_period)
    last_state_trainee = featurize(last_state_trainee, device)
    last_state_opponent = last_state[opponent_id]

    num_boards = last_state_trainee.size()[1]
    board_size = last_state_trainee.size()[2]

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    print("Action Space: {}".format(n_actions))

    model_file = os.path.join(store_mode['MODEL_PATH'], store_mode['MODEL_NAME'])
    best_model_file = os.path.join(store_mode['MODEL_PATH'], store_mode['BEST_MODEL_NAME'])
    network = path_to_instance_simple(algorithm_settings["NETWORK"])
    policy_net = network(board_size, num_boards, n_actions).to(device)
    policy_net.eval()
    if general_settings['LOAD_MODEL'] == "LATEST":
        print("Load model: {}".format(model_file))
        policy_net.load_state_dict(torch.load(model_file, map_location=general_settings['DEVICE']))
    elif general_settings['LOAD_MODEL'] == "BEST":
        print("Load model: {}".format(best_model_file))
        policy_net.load_state_dict(torch.load(best_model_file, map_location=general_settings['DEVICE']))
    else:
        print("Create directory: {}", store_mode['MODEL_PATH'])
        os.makedirs(store_mode['MODEL_PATH'])

    target_net = network(board_size, num_boards, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # notifies net that we are in evaluation mode (batchnorm or dropout layers)

    optimizer = optim.Adam(policy_net.parameters(), lr=algorithm_settings["LR"])
    memory = ReplayMemory(algorithm_settings['EXPERIENCE_REPLAY_SIZE'])

    episode_count = 1
    episode_reward = 0
    episode_steps = 0
    averaged_episode_steps = []
    averaged_episode_rewards = []
    eps_history = []
    loss_sum = 0
    best_model_reward = -100

    while episode_count <= algorithm_settings['NUM_EPISODES']:
        if render_options["RENDER_ALL"]:
            env.render()

        last_action = select_action(last_state_trainee, n_actions, episode_count)
        opponent_action = opponent.act(last_state_opponent, env.action_space)
        actions = [0,0]
        real_action = postprocess_action(action_filter, jitter_filter, last_action, env, agent_pos, trainee)
        actions[trainee_id] = real_action
        actions[opponent_id] = opponent_action
        current_state, reward, terminal, info = env.step(actions)
        reward = float(reward[trainee_id])
        current_state_trainee = current_state[trainee_id]
        current_state_opponent = current_state[opponent_id]
        reward = reward_modifier.modify_reward(current_state_trainee, reward)
        # action filter for next step
        action_filter = ActionFilter(current_state_trainee)
        episode_reward += reward
        episode_steps += 1
        agent_pos = current_state_trainee['position']

        current_state_trainee = featurize(current_state_trainee, device)
        reward = torch.tensor([reward], device=device)
        terminal = torch.tensor([terminal], device=device, dtype=torch.bool)
        memory.push(last_state_trainee, last_action, reward, current_state_trainee, terminal)

        loss = optimize_model()
        loss_sum += loss.item()

        if terminal:
            episode_count += 1
            # env.render()
            env.close()
            env, trainee, trainee_id, opponent, opponent_id = curriculum.get_env(episode_count)
            last_state = env.reset()
            last_state_trainee = last_state[trainee_id]
            action_filter = ActionFilter(last_state_trainee)
            jitter_filter.reset()
            last_state_opponent = last_state[opponent_id]
            reward_modifier = RewardModifier(last_state_trainee, curriculum.current_period)
            last_state_trainee = featurize(last_state_trainee, device)
            # print("Epsiode count: {}, R: {}, Steps: {}".format(episode_count, reward, episode_steps))
        else:
            last_state_trainee = current_state_trainee
            last_state_opponent = current_state_opponent

        if terminal:
            # Update the target network, copying all weights and biases in DQN
            if episode_count % algorithm_settings['TARGET_UPDATE'] == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if episode_count % stats['PRINT_STATS'] == 0:
                avg_steps = episode_steps / stats['PRINT_STATS']
                averaged_episode_steps.append(avg_steps)
                avg_reward = episode_reward / stats['PRINT_STATS']
                averaged_episode_rewards.append(avg_reward)
                eps_history.append(algorithm_settings['EPS'])
                episode_steps = 0
                episode_reward = 0

                print("Episode: {}; EPS: {}; Avg. Steps: {}; Mem. len: {}; Avg. Reward: {}".format(
                    episode_count, algorithm_settings['EPS'], avg_steps, len(memory), avg_reward))

                if general_settings["WARMUP"]:
                    plot_warmup(stats['PRINT_STATS'], stats['FIGURE_STATS'], averaged_episode_rewards,\
                                averaged_episode_steps, eps_history)
                else:
                    plot(stats['PRINT_STATS'], stats['FIGURE_STATS'], averaged_episode_rewards,\
                         averaged_episode_steps, eps_history)

            if episode_count % stats['UPDATE_TENSORBOARD'] == 0 and \
                len(memory) > algorithm_settings['MIN_EXPERIENCE_REPLAY_SIZE']:

                average_loss = loss_sum/stats['UPDATE_TENSORBOARD']
                loss_sum = 0
                writer.add_scalar(tag="training/reward",
                                  scalar_value=averaged_episode_rewards[-1],
                                  global_step=episode_count)

                writer.add_scalar(tag="training/steps",
                                  scalar_value=averaged_episode_steps[-1],
                                  global_step=episode_count)

                writer.add_scalar(tag="training/eps",
                                  scalar_value=algorithm_settings['EPS'],
                                  global_step=episode_count)

                writer.add_scalar(tag="training/episode_loss",
                                  scalar_value=average_loss,
                                  global_step=episode_count)


                # parameters
                for i, param in enumerate(policy_net.parameters()):
                    writer.add_histogram(tag=f'policy_net/gradients_{i}', values=param.grad.cpu(),
                                        global_step=episode_count)

                # Add weights as arrays to tensorboard
                for i, param in enumerate(policy_net.parameters()):
                    writer.add_histogram(tag=f'policy_net/param_{i}', values=param.cpu(),
                                         global_step=episode_count)

                # Add weights as arrays to tensorboard
                for i, param in enumerate(target_net.parameters()):
                    writer.add_histogram(tag=f'target_net/param_{i}', values=param.cpu(),
                                         global_step=episode_count)

            if episode_count % store_mode['SAVE_MODEL'] == 0:
                torch.save(policy_net.state_dict(), model_file)
                if averaged_episode_rewards[-1] > best_model_reward:
                    best_model_reward = averaged_episode_rewards[-1]
                    torch.save(policy_net.state_dict(), best_model_file)

    env.close()
    print('Completed Training')

    print("Store results in Pickle file")
    results_dict = {'avg_reward': averaged_episode_rewards, 'avg_steps': averaged_episode_steps,
                    'eps_history': eps_history}
    with open(os.path.join(stats["RESULTS_PATH"], stats["PICKLE_RESULT"]), 'wb') as f:
        pickle.dump(results_dict, f)






if __name__ == "__main__":
    train()

