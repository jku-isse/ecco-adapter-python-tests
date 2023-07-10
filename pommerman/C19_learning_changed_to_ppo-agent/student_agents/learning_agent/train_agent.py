import torch
import os
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import pickle
from pommerman.constants import Action
from pommerman import constants

from ppo import PPO
from learning_agent.github_source.storage import RolloutStorage
from learning_agent.net_input import featurize
from helper import create_random_env, parse_conf, path_to_instance_simple
from reward_modifier import RewardModifier
from plot import plot, plot_warmup
from action_filter import ActionFilter
from jitter_filter import JitterFilter
from curriculum_learning import Curriculum


def postprocess_action(action_filter, jitter_filter, last_action, env, agent_pos, trainee):
    real_action = last_action
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

    # create env
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
    num_processes = 1  # currently only the version with one single process is available

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    print("Action Space: {}".format(n_actions))

    model_file = os.path.join(store_mode['MODEL_PATH'], store_mode['MODEL_NAME'])
    best_model_file = os.path.join(store_mode['MODEL_PATH'], store_mode['BEST_MODEL_NAME'])
    network = path_to_instance_simple(algorithm_settings["NETWORK"])
    model = network(board_size, num_boards, n_actions)
    model.to(device)
    model.eval()
    if general_settings['LOAD_MODEL'] == "LATEST":
        print("Load model: {}".format(model_file))
        model.load_state_dict(torch.load(model_file, map_location=general_settings['DEVICE']))
    elif general_settings['LOAD_MODEL'] == "BEST":
        print("Load model: {}".format(best_model_file))
        model.load_state_dict(torch.load(best_model_file, map_location=general_settings['DEVICE']))
    else:
        print("Create directory: {}", store_mode['MODEL_PATH'])
        os.makedirs(store_mode['MODEL_PATH'])

    agent = PPO(
        model,
        algorithm_settings["CLIP_PARAM"],
        algorithm_settings["PPO_EPOCH"],
        algorithm_settings["NUM_MINI_BATCH"],
        algorithm_settings["VALUE_LOSS_COEF"],
        algorithm_settings["ENTROPY_COEF"],
        lr=algorithm_settings["LR"],
        eps=algorithm_settings["EPS"],
        max_grad_norm=algorithm_settings["MAX_GRAD_NORM"],
        use_clipped_value_loss=True)

    # in rollout we want to store the already featurized observations
    obs_shape = (num_boards, board_size, board_size)

    rollouts = RolloutStorage(algorithm_settings["NUM_STEPS"], num_processes, obs_shape)
    rollouts.obs[0].copy_(last_state_trainee)
    rollouts.to(device)

    episode_count = 1
    episode_reward = 0
    episode_steps = 0
    averaged_episode_steps = []
    averaged_episode_rewards = []
    best_model_reward = -100
    episode_step_count = 0
    value_loss, action_loss, dist_entropy = (0,0,0)

    # training loop
    while episode_count < algorithm_settings["NUM_EPISODES"]:
        for step in range(algorithm_settings["NUM_STEPS"]):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = model(rollouts.obs[step])

            # apllying filters
            last_action = action.item()
            real_action = postprocess_action(action_filter, jitter_filter, last_action, env, agent_pos, trainee)

            # observe move from opponent
            opponent_action = opponent.act(last_state_opponent, env.action_space)
            actions = [None]*2
            actions[trainee_id] = real_action
            actions[opponent_id] = opponent_action
            current_state, reward, terminal, info = env.step(actions)
            episode_step_count += 1
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
            # If done then clean the history of observations.
            masks = torch.FloatTensor([0.0] if terminal else [1.0])
            bad_masks = torch.FloatTensor([[0.0] if episode_step_count >= constants.MAX_STEPS else [1.0]])


            if terminal:
                episode_step_count = 0
                episode_count += 1
                env.close()
                env, trainee, trainee_id, opponent, opponent_id = curriculum.get_env(episode_count)
                last_state = env.reset()
                last_state_trainee = last_state[trainee_id]
                action_filter = ActionFilter(last_state_trainee)
                jitter_filter.reset()
                last_state_opponent = last_state[opponent_id]
                reward_modifier = RewardModifier(last_state_trainee, curriculum.current_period)
                last_state_trainee = featurize(last_state_trainee, device)
            else:
                last_state_trainee = current_state_trainee
                last_state_opponent = current_state_opponent

            rollouts.insert(last_state_trainee, action.squeeze(0), action_log_prob.squeeze(0), value.squeeze(0),
                            reward, masks, bad_masks)

            if render_options["RENDER_ALL"]:
                env.render()

            if terminal:
                if episode_count % stats['PRINT_STATS'] == 0:
                    avg_steps = episode_steps / stats['PRINT_STATS']
                    averaged_episode_steps.append(avg_steps)
                    avg_reward = episode_reward / stats['PRINT_STATS']
                    averaged_episode_rewards.append(avg_reward)
                    episode_steps = 0
                    episode_reward = 0

                    print("Episode: {}; Avg. Steps: {}; Avg. Reward: {}".format(
                        episode_count, avg_steps, avg_reward))

                    if general_settings["WARMUP"]:
                        plot_warmup(stats['PRINT_STATS'], stats['FIGURE_STATS'], averaged_episode_rewards,
                                    averaged_episode_steps)
                    else:
                        plot(algorithm_settings['PERIOD_LENGTH'], stats['PRINT_STATS'], stats['FIGURE_STATS'],
                             averaged_episode_rewards, averaged_episode_steps)

                if episode_count % stats['UPDATE_TENSORBOARD'] == 0 and episode_count > stats['UPDATE_TENSORBOARD']:
                    writer.add_scalar(tag="training/reward",
                                      scalar_value=averaged_episode_rewards[-1],
                                      global_step=episode_count)

                    writer.add_scalar(tag="training/steps",
                                      scalar_value=averaged_episode_steps[-1],
                                      global_step=episode_count)

                    writer.add_scalar(tag="training/value_loss",
                                      scalar_value=value_loss,
                                      global_step=episode_count)

                    writer.add_scalar(tag="training/action_loss",
                                      scalar_value=action_loss,
                                      global_step=episode_count)

                    writer.add_scalar(tag="training/dist_entropy",
                                      scalar_value=dist_entropy,
                                      global_step=episode_count)

                    # parameters
                    for i, param in enumerate(model.parameters()):
                        writer.add_histogram(tag=f'policy_net/gradients_{i}', values=param.grad.cpu(),
                                             global_step=episode_count)

                    # Add weights as arrays to tensorboard
                    for i, param in enumerate(model.parameters()):
                        writer.add_histogram(tag=f'policy_net/param_{i}', values=param.cpu(),
                                             global_step=episode_count)

                    # Add weights as arrays to tensorboard
                    for i, param in enumerate(model.parameters()):
                        writer.add_histogram(tag=f'target_net/param_{i}', values=param.cpu(),
                                             global_step=episode_count)

                if episode_count % store_mode['SAVE_MODEL'] == 0:
                    torch.save(model.state_dict(), model_file)
                    if averaged_episode_rewards[-1] > best_model_reward:
                        best_model_reward = averaged_episode_rewards[-1]
                        torch.save(model.state_dict(), best_model_file)

        with torch.no_grad():
            next_value = model.get_value(rollouts.obs[-1]).detach()

        rollouts.compute_returns(next_value, algorithm_settings["USE_GAE"], algorithm_settings["GAMMA"],
                                 algorithm_settings["GAE_LAMBDA"], algorithm_settings["USE_PROPER_TIME_LIMITS"])

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

    env.close()
    print('Completed Training')

    print("Store results in Pickle file")
    results_dict = {'avg_reward': averaged_episode_rewards, 'avg_steps': averaged_episode_steps}
    with open(os.path.join(stats["RESULTS_PATH"], stats["PICKLE_RESULT"]), 'wb') as f:
        pickle.dump(results_dict, f)


if __name__ == "__main__":
    train()
