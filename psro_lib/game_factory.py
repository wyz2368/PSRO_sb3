import pyspiel
from shimmy import OpenSpielCompatibilityV0
from pettingzoo.classic import tictactoe_v3
from pettingzoo.classic import leduc_holdem_v4
from pettingzoo.classic import rps_v2

from applications.mixnet_homo.homo_mixnet_env import Mixnet_env
from applications.mixnet.mixnet_env import Mixnet_env as Mixnet_env_hetero


def load_spiel_env(game_name):
    env = pyspiel.load_game(game_name)
    env = OpenSpielCompatibilityV0(env)
    return env


GAMES = {
    "mixnet_hetero": Mixnet_env_hetero(),
    "mixnet_homo": Mixnet_env(),
    "rps_v2": rps_v2.env(),
    "tictactoe_v3": tictactoe_v3.env(),
    "leduc_holdem_v4": leduc_holdem_v4.env(),
    "leduc_poker": load_spiel_env("leduc_poker"),
    "kuhn_poker": load_spiel_env("kuhn_poker"),
    "blotto": load_spiel_env("blotto")
}

def get_env_factory(game_name):
    def get_env():
        env = GAMES[game_name]
        return env
    return get_env