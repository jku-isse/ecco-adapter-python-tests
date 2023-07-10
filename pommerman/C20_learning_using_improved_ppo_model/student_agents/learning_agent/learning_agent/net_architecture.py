import torch.nn as nn
import numpy as np
from torch import distributions

from github_source.util import init
from github_source.distributions import Categorical

class SimpleActorCritic(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_outputs):
        super(SimpleActorCritic, self).__init__()

        # weight initalisation function
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

        # choose correct distribution - for pommerman it is discrete
        self.dist = Categorical(hidden_size, num_outputs)


    def forward(self, inputs):
        critic_value = self.critic_linear(self.critic(inputs))
        hidden_actor = self.actor(inputs)

        # get action distribution
        dist = self.dist(hidden_actor)
        action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return critic_value, action, action_log_probs

    def get_value(self, inputs):
        value, _, _ = self(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        critic_value = self.critic_linear(self.critic(inputs))
        hidden_actor = self.actor(inputs)

        # get action distribution
        dist = self.dist(hidden_actor)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return critic_value, action_log_probs, dist_entropy


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)



class CNNActorCritic(nn.Module):
    def __init__(self, board_size, num_boards, num_actions):
        super(CNNActorCritic, self).__init__()

        cnn_out_channels = 64

        self.shared = nn.Sequential(
            nn.Conv2d(in_channels=num_boards, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=cnn_out_channels, kernel_size=3, stride=1),
            nn.ReLU(),
            Flatten()
        )

        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        out_size = conv2d_size_out(conv2d_size_out(conv2d_size_out(board_size)))

        self.critic = nn.Sequential(
            nn.Linear(out_size * out_size * cnn_out_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(out_size * out_size * cnn_out_channels, 256),
            nn.ReLU()
        )

        # choose correct distribution - for pommerman it is discrete (categorical)
        self.dist = Categorical(256, num_actions)

        self.train()

    def forward(self, inputs):
        shared_out = self.shared(inputs)
        hidden_actor = self.actor(shared_out)
        critic_value = self.critic(shared_out)

        # get action distribution
        dist = self.dist(hidden_actor)
        action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return critic_value, action, action_log_probs

    def get_value(self, inputs):
        value, _, _ = self(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        shared_out = self.shared(inputs)
        hidden_actor = self.actor(shared_out)
        critic_value = self.critic(shared_out)

        # get action distribution
        dist = self.dist(hidden_actor)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return critic_value, action_log_probs, dist_entropy


class CNNActorCriticSoftmax(nn.Module):
    def __init__(self, board_size, num_boards, num_actions):
        super(CNNActorCriticSoftmax, self).__init__()

        cnn_out_channels = 64

        self.shared = nn.Sequential(
            nn.Conv2d(in_channels=num_boards, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=cnn_out_channels, kernel_size=3, stride=1),
            nn.ReLU(),
            Flatten()
        )

        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        out_size = conv2d_size_out(conv2d_size_out(conv2d_size_out(board_size)))

        self.critic = nn.Sequential(
            nn.Linear(out_size * out_size * cnn_out_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(out_size * out_size * cnn_out_channels, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
            nn.Softmax(dim=-1)
        )

        # choose correct distribution - for pommerman it is discrete (categorical)
        # self.dist = Categorical(256, num_actions)
        self.train()

    def forward(self, inputs):
        shared_out = self.shared(inputs)
        action_probs = self.actor(shared_out)
        critic_value = self.critic(shared_out)

        # get action distribution
        dist = distributions.Categorical(action_probs)
        action = dist.sample()

        action_log_probs = dist.log_prob(action)

        return critic_value, action, action_log_probs

    def get_value(self, inputs):
        value, _, _ = self(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        shared_out = self.shared(inputs)
        action_probs = self.actor(shared_out)
        critic_value = self.critic(shared_out)

        # get action distribution
        dist = distributions.Categorical(action_probs)

        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy().mean()

        return critic_value, action_log_probs, dist_entropy


class CNNActorCriticBorealis(nn.Module):
    def __init__(self, board_size, num_boards, num_actions):
        super(CNNActorCriticBorealis, self).__init__()

        # weight initalisation function
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.shared = nn.Sequential(
            init_(nn.Conv2d(in_channels=num_boards, out_channels=64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            init_(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            init_(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            init_(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU()
        )

        lin_init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        cnn_out_channels = 1
        self.cnn_out_size = board_size * board_size * cnn_out_channels

        self.actor = nn.Sequential(
            init_(nn.Conv2d(in_channels=64, out_channels=cnn_out_channels, kernel_size=1, stride=1)),
            nn.ReLU(),
            Flatten()
        )

        self.critic = nn.Sequential(
            init_(nn.Conv2d(in_channels=64, out_channels=cnn_out_channels, kernel_size=1, stride=1)),
            nn.ReLU(),
            Flatten()
        )

        self.critic_value = nn.Sequential(
            lin_init_(nn.Linear(self.cnn_out_size, 1)),
            nn.Tanh()
        )

        self.actor_softmax = nn.Sequential(
            lin_init_(nn.Linear(self.cnn_out_size, num_actions)),
            nn.Softmax(dim=1)
        )

        self.train()

    def forward(self, inputs):
        shared_out = self.shared(inputs)
        action_probs = self.actor_softmax(self.actor(shared_out))
        critic_value = self.critic_value(self.critic(shared_out))

        # get action distribution
        dist = distributions.Categorical(action_probs)
        action = dist.sample()

        action_log_probs = dist.log_prob(action)

        return critic_value, action, action_log_probs

    def get_value(self, inputs):
        value, _, _ = self(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        shared_out = self.shared(inputs)
        action_probs = self.actor_softmax(self.actor(shared_out))
        critic_value = self.critic_value(self.critic(shared_out))

        # get action distribution
        dist = distributions.Categorical(action_probs)

        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy().mean()

        return critic_value, action_log_probs, dist_entropy

