import os
import sys

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Get the grandparent directory
grandparent_dir = os.path.dirname(parent_dir)

# Add the grandparent directory to the sys.path
sys.path.append(grandparent_dir)



"""
Example running PSRO.
"""

import time
import datetime
import os
from absl import app
from absl import flags
import numpy as np
import pickle
import random
from tensorboardX import SummaryWriter

from psro_lib.psro import PSROSolver
from psro_lib.game_factory import get_env_factory
from psro_lib.rl_agent_sb3.rl_factory import generate_agent_class
from psro_lib.rl_agent_sb3.rl_oracle import RLOracle, freeze_all
from psro_lib.utils import init_logger, save_pkl
from psro_lib.eval_utils import regret_of_last_iter, mixed_strategy_payoff
from solution_solvers.nash_solver.pygambit_solver import pygbt_solve_matrix_games


FLAGS = flags.FLAGS
# Game-related
flags.DEFINE_string("game_name", "mixnet_hetero", "Game names: kuhn_poker, tictactoe_v3, leduc_holdem_v4")
flags.DEFINE_integer("n_players", 2, "The number of players.")

# PSRO related
flags.DEFINE_string("meta_strategy_method", "nash",
                    "Name of meta strategy computation method.")
flags.DEFINE_integer("sims_per_entry", 1,
                     ("Number of simulations to run to estimate each element"
                      "of the game outcome matrix."))

flags.DEFINE_integer("gpsro_iterations", 5,
                     "Number of training steps for GPSRO.")
flags.DEFINE_bool("symmetric_game", False, "Whether to consider the current "
                                           "game as a symmetric game.")

# Rectify options
flags.DEFINE_string("rectifier", "",
                    "(No filtering), 'rectified' for rectified.")

# General (RL) agent parameters
flags.DEFINE_string("oracle_type", "DQN", "DQN, PPO, MaskablePPO (MaskableActorCriticPolicy)")
flags.DEFINE_integer("number_training_episodes", int(500), "Number training (default 1e4) " ############
                                                           "episodes per RL policy. Used for PG and DQN")
flags.DEFINE_float("self_play_proportion", 0.0, "Self play proportion")
flags.DEFINE_integer("hidden_layer_size", 256, "Hidden layer size")
flags.DEFINE_integer("hidden_layers", 4, "Hidden layer size")
flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_float("sigma", 0.0, "Policy copy noise (Gaussian Dropout term).")
flags.DEFINE_string("optimizer_str", "adam", "'adam' or 'sgd'")
flags.DEFINE_integer("n_hidden_layers", 4, "# of hidden layers")
flags.DEFINE_float("dqn_learning_rate", 5e-2, "DQN learning rate.")


# General
flags.DEFINE_string("root_result_folder", 'root_result_psro', "root directory of saved results")
flags.DEFINE_integer("seed", None, "Seed.")
flags.DEFINE_integer("verbose", 1, "Enables verbose printing and profiling.")
flags.DEFINE_bool("dummy_env", True, "Enables dummy env otherwise subproc env")


def init_oracle(env):

    # Return the RL policy from SB3. Currently, assume DQN.
    agent_class = generate_agent_class(agent_name=FLAGS.oracle_type)

    agent_kwargs = {
        "policy": "MlpPolicy",
        "hidden_layers_sizes": FLAGS.hidden_layer_size,
        "hidden_layers": FLAGS.hidden_layers,
        "batch_size": FLAGS.batch_size,
        "learning_rate": FLAGS.dqn_learning_rate,
    }


    # Create BR oracle.
    oracle = RLOracle(env=env,
                      best_response_class=agent_class,
                      best_response_kwargs=agent_kwargs,
                      total_timesteps=FLAGS.number_training_episodes,
                      sigma=FLAGS.sigma,
                      verbose=0)

    agents = oracle.generate_new_policies()
    freeze_all(agents)

    # agents = [RandomPolicy(), RandomPolicy()]

    return oracle, agents


def save_at_termination(solver, file_for_meta_game):
    with open(file_for_meta_game, 'wb') as f:
        pickle.dump(solver.get_meta_game(), f)


def gpsro_looper(env, oracle, agents, writer, checkpoint_dir=None, seed=None):
    """Initializes and executes the GPSRO training loop."""
    sample_from_marginals = True

    # Logging important information
    logger = init_logger(logger_name=__name__, checkpoint_dir=checkpoint_dir)
    logger.info("Game name: {}".format(FLAGS.game_name))
    logger.info("Number of players: {}".format(FLAGS.n_players))
    logger.info("Meta strategy method: {}".format(FLAGS.meta_strategy_method))
    logger.info("Oracle type: {}".format(FLAGS.oracle_type))

    # Create a PSRO solver.
    g_psro_solver = PSROSolver(env,
                            oracle,
                            initial_policies=agents,
                            rectifier=FLAGS.rectifier,
                            sims_per_entry=FLAGS.sims_per_entry,
                            meta_strategy_method=FLAGS.meta_strategy_method,
                            prd_iterations=int(1e5),  # 50000
                            prd_gamma=1e-3,
                            sample_from_marginals=sample_from_marginals,
                            symmetric_game=FLAGS.symmetric_game,
                            checkpoint_dir=checkpoint_dir,
                            dummy_env=FLAGS.dummy_env)

    start_time = time.time()
    for gpsro_iteration in range(1, FLAGS.gpsro_iterations + 1):
        if FLAGS.verbose:
            logger.info("\n===========================\n")
            logger.info("Iteration : {}".format(gpsro_iteration))
            logger.info("Time so far: {}".format(time.time() - start_time))
        g_psro_solver.iteration(seed=seed)
        meta_game = g_psro_solver.get_meta_game()
        meta_probabilities = g_psro_solver.get_meta_strategies()
        nash_meta_probabilities = g_psro_solver.get_nash_strategies()

        if FLAGS.verbose:
            logger.info("Meta game : {}".format(meta_game))
            logger.info("Probabilities : {}".format(meta_probabilities))
            logger.info("Nash Probabilities : {}".format(nash_meta_probabilities))

        if gpsro_iteration == FLAGS.gpsro_iterations:
            save_at_termination(solver=g_psro_solver, file_for_meta_game=checkpoint_dir + '/meta_game.pkl')

        # Regret Measure
        if gpsro_iteration > 1:
            regrets = regret_of_last_iter(meta_game, stored_meta_probabilities, stored_expected_payoffs)
            logger.info("Regrets : {}".format(regrets))
            writer.add_scalar('Sum_of_regrets', sum(regrets), gpsro_iteration)

        expected_payoffs = mixed_strategy_payoff(meta_game, nash_meta_probabilities)
        stored_meta_probabilities = nash_meta_probabilities
        stored_expected_payoffs = expected_payoffs


        save_pkl(checkpoint_dir + "/meta_game.pkl", meta_game)

    meta_game = g_psro_solver.get_meta_game()
    all_ne = pygbt_solve_matrix_games(meta_game, method="enummixed", mode="all")
    for ne in all_ne:
        logger.info("===== Find all NE =====")
        expected_payoffs = mixed_strategy_payoff(meta_game, ne)
        logger.info("Nash Probabilities : {}".format(ne))
        logger.info("Expected payoff : {}".format(expected_payoffs))



def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Set up the seed.
    if FLAGS.seed is None:
        seed = np.random.randint(low=0, high=1e5)
    else:
        seed = FLAGS.seed
    np.random.seed(seed)
    random.seed(seed)

    # Load game. This should be adaptive to different environments.
    env = get_env_factory(FLAGS.game_name)()
    # print("NNNNNN:", env.metadata["name"])
    env.reset()

    # Set up working directory.
    if not os.path.exists(FLAGS.root_result_folder):
        os.makedirs(FLAGS.root_result_folder)

    checkpoint_dir = FLAGS.game_name
    checkpoint_dir += checkpoint_dir + "_oracle_" + FLAGS.oracle_type + '_se_' + str(seed) + '_' + datetime.datetime.now().strftime(
        '%Y-%m-%d_%H-%M-%S')
    checkpoint_dir = os.path.join(os.getcwd(), FLAGS.root_result_folder, checkpoint_dir)

    writer = SummaryWriter(logdir=checkpoint_dir + '/log')

    # Initialize oracle and agents
    oracle, agents = init_oracle(env=env)

    gpsro_looper(env, oracle, agents, writer, checkpoint_dir=checkpoint_dir, seed=seed)

    writer.close()


if __name__ == "__main__":
    app.run(main)
