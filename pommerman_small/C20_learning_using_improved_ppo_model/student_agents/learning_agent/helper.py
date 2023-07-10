import importlib
import argparse
import os
import json
import shutil

import pommerman
from pommerman.agents import SimpleAgent, PytorchAgent

def create_random_env():
    trainee = PytorchAgent()
    opponent = SimpleAgent()
    trainee_id = 0
    opponent_id = 1
    agents = [trainee, opponent]
    env = pommerman.make('PommeFFACompetition-v0', agents, game_state_file=os.path.join('arenas', 'arena-09.json'))
    env.set_agents(agents)
    env.set_training_agent(trainee.agent_id)
    return env, trainee, trainee_id, opponent, opponent_id

def path_to_instance_simple(network):
    print("Using network: {}".format(network))
    parts = network.split('.')
    module = ".".join(parts[:-1])
    module = importlib.import_module(module)
    network = getattr(module, parts[-1])
    return network


def parse_conf():
    # parse command line args
    parser = argparse.ArgumentParser(description="Config-File")

    parser.add_argument('config', type=str, nargs='?', default="config.json")

    args = parser.parse_args()
    path_to_config = args.config

    if not os.path.isfile(path_to_config):
        raise FileNotFoundError("The given config file could not be found")

    with open(path_to_config, 'r') as f:
        config = json.load(f)

    general_settings = config["general"]
    algorithm_settings = config["algorithm_settings"]
    store_mode = config["store_model"]
    render_options = config["render_options"]
    stats = config["stats"]

    if general_settings["DEBUG_MODE"]:
        print("Running in DEBUG mode")
        general_settings["LOAD_MODEL"] = "LATEST"
        general_settings["DEVICE"] = "cpu"
        general_settings["LOAD_ARCHITECTURE"] = "cpu"
        render_options["RENDER_ALL"] = True

    if general_settings["WARMUP"]:
        warmup_settings = algorithm_settings["WARMUP_SETTINGS"]
        algorithm_settings["NUM_EPISODES"] = warmup_settings["NUM_EPISODES"]
        algorithm_settings["PERIOD_LENGTH"] = [warmup_settings["NUM_EPISODES"], 0, 0, 0]

    experiment_folder = "EXPERIMENT_" + general_settings["EXPERIMENT_NUMBER"]
    store_mode["MODEL_PATH"] = os.path.join(experiment_folder, store_mode["MODEL_PATH"])
    stats["RESULTS_PATH"] = os.path.join(experiment_folder, stats["RESULTS_PATH"])
    stats["FIGURE_STATS"] = os.path.join(experiment_folder, stats["FIGURE_STATS"])

    if general_settings["LOAD_MODEL"] == 'NONE' and os.path.exists(experiment_folder):
        answer = input("{} already exists. Overwrite ? (y/n)".format(experiment_folder))
        while not(answer == "n" or answer == "y"):
            answer = input("Type y (yes) or n (no -> exit)")
        if answer == "n":
            exit(0)
        else:
            shutil.rmtree(experiment_folder)
            os.makedirs(experiment_folder)

    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)

    shutil.copy(path_to_config, experiment_folder)

    return general_settings, algorithm_settings, store_mode, render_options, stats

def path_to_instance_simple(network):
    print("Using network: {}".format(network))
    parts = network.split('.')
    module = ".".join(parts[:-1])
    module = importlib.import_module(module)
    network = getattr(module, parts[-1])
    return network
