"""An example to show how to set up an pommerman game programmatically"""

import pommerman
from pommerman import agents
from pommerman.constants import Result
from student_agents.learning_agent.learning_agent import learning_agent


def main():
    """Simple function to bootstrap a game."""
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    # Create a set of agents
    agent_list = [
        agents.SimpleAgent(),
        learning_agent.LearningAgent(),
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    # Run the episodes just like OpenAI Gym
    for i_episode in range(3):
        state = env.reset()
        done = False
        while not done:
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
        print('Episode {} finished'.format(i_episode))

        if info["result"] == Result.Win.value:
            print("winner %d" % info["winner"])
        else:
            print("Draw!")

    env.close()


if __name__ == '__main__':
    main()
