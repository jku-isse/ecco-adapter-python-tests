import torch
import os
import pkg_resources

from pommerman import agents

from . import net_input
from . import net_architecture


# an example on how the trained agent can be used within the tournament
class LearningAgent(agents.BaseAgent):
    def __init__(self, *args, **kwargs):
        super(LearningAgent, self).__init__(*args, **kwargs)
        self.device = torch.device("cpu")  # you only have access to cpu during the tournament
        data_path = pkg_resources.resource_filename('//remotestorageurl/pommerman', 'dqn1')
        model_file = os.path.join(data_path, 'model.pt')

        # loading the trained neural network model
        self.model = net_architecture.DQN(board_size=11, num_boards=7, num_actions=6)
        self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        self.model.eval()

    def act(self, obs, action_space):
        # the learning agent uses the neural net to find a move
        # the observation space has to be featurized before it is fed to the model
        obs_featurized = net_input.featurize_simple(obs, self.device)
        with torch.no_grad():
            action = self.model(obs_featurized).max(1)[1]  # take highest rated move
        return action.item()
