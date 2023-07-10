import torch
import torch.nn as nn

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)

class QN(nn.Module):
    def __init__(self, board_size, num_boards, num_actions):
        super(QN, self).__init__()
        out_channels = 16
        self.conv = nn.Conv2d(in_channels=num_boards, out_channels=out_channels, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        out_size = conv2d_size_out(11)

        self.activation = nn.ReLU()
        self.head = nn.Linear(out_size * out_size * out_channels, num_actions)

        torch.nn.init.xavier_uniform_(self.conv.weight)
        torch.nn.init.xavier_uniform_(self.head.weight)

    def forward(self, x):
        x = self.activation(self.conv(x))
        x = self.head(x.view(x.size(0), -1))
        return x


class DQN(nn.Module):
    def __init__(self, board_size, num_boards, num_actions):
        super(DQN, self).__init__()
        cnn_out_channels = 64

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=num_boards, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=cnn_out_channels, kernel_size=3, stride=1),
            nn.ReLU()
        )

        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        out_size = conv2d_size_out(conv2d_size_out(conv2d_size_out(board_size)))

        self.value_head = nn.Sequential(
            nn.Linear(out_size * out_size * cnn_out_channels, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

        self.conv.apply(weights_init)
        self.value_head.apply(weights_init)

    def forward(self, state):
        x = self.conv(state)
        qvals = self.value_head(x.view(x.size(0), -1))
        return qvals

class DQN_(nn.Module):
    def __init__(self, board_size, num_boards, num_actions):
        super(DQN_, self).__init__()
        cnn_out_channels = 2
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=num_boards, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=cnn_out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        linear_input_size = board_size * board_size * cnn_out_channels
        self.head = nn.Linear(linear_input_size, num_actions)

        self.conv_net.apply(weights_init)
        self.head.apply(weights_init)

    def forward(self, x):
        x = self.conv_net(x)  # batch_size x num_channels x board_size x board_size
        x = self.head(x.view(x.size(0), -1)) # batch_size x 2 x board_size x board_size
        return x # batch_size * 6