import os
from typing import Optional, Tuple, Any, Literal, Protocol, Self, TypeVar, cast, overload

import gymnasium
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.policy.multiagent.mapolicy import MapTrainingStats, MAPRolloutBatchProtocol
from typing import Any
from tianshou.data import Batch, ReplayBuffer
from tianshou.data.batch import BatchProtocol, IndexType
from tianshou.data.types import ActBatchProtocol, ObsBatchProtocol, RolloutBatchProtocol
from tianshou.policy import BasePolicy
from tianshou.policy.base import TLearningRateScheduler, TrainingStats
from applications.mixnet_homo.homo_mixnet_env import Mixnet_env

from pettingzoo.classic import tictactoe_v3

from copy import deepcopy


def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    return PettingZooEnv(tictactoe_v3.env())


def freeze(policy):
    nn = policy.model
    for p in nn.parameters():
        p.requires_grad = True


def print_params(policy):
    nn = policy.model
    #     print(nn)
    for p in nn.parameters():
        print(p.data)
        break


def _get_agents(
        agent_learn: Optional[BasePolicy] = None,
        agent_opponent: Optional[BasePolicy] = None,
        optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = _get_env()
    print("Agents:", env.agents)
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gymnasium.spaces.Dict)
        else env.observation_space
    )
    if agent_learn is None:
        # model
        print(observation_space.shape or observation_space.n)
        print(env.action_space.shape or env.action_space.n)
        net = Net(
            state_shape=observation_space.shape or observation_space.n,
            action_shape=env.action_space.shape or env.action_space.n,
            hidden_sizes=[128, 128, 128, 128],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        if optim is None:
            optim = torch.optim.Adam(net.parameters(), lr=1e-4)
        agent_learn = DQNPolicy(
            model=net,
            optim=optim,
            action_space=env.action_space,
            discount_factor=0.9,
            estimation_step=3,
            target_update_freq=320,
        )

    if agent_opponent is None:
        agent_opponent = RandomPolicy(action_space=env.action_space)
    #         agent_opponent = deepcopy(agent_learn)

    agents = [agent_opponent, agent_learn]
    #     agents = [agent_learn, agent_opponent]
    policy = MultiAgentPolicyManager(policies=agents, env=env)
    #     policy = MultiAgentPolicyManager_PSRO(policies=agents, env=env, learning_players_id=1)

    #     policy.replace_policy(RandomPolicy(), "player_1")
    #     policy.replace_policy(RandomPolicy(), 0)

    return policy, optim, env.agents


from overrides import override


class NoneTrainingStats(TrainingStats):
    pass


class CustomizedMapTrainingStats(TrainingStats):
    def __init__(
            self,
            agent_id_to_stats: dict[str | int, TrainingStats],
            learning_players_id,
            train_time_aggregator: Literal["min", "max", "mean"] = "max",
    ) -> None:
        self._agent_id_to_stats = agent_id_to_stats
        train_times = [agent_stats.train_time for agent_stats in agent_id_to_stats.values()]
        match train_time_aggregator:
            case "max":
                aggr_function = max
            case "min":
                aggr_function = min
            case "mean":
                aggr_function = np.mean  # type: ignore
            case _:
                raise ValueError(
                    f"Unknown {train_time_aggregator=}",
                )
        self.train_time = aggr_function(train_times)
        self.learning_players_id = learning_players_id
        self.smoothed_loss = {}

    @override
    def get_loss_stats_dict(self) -> dict[str, float]:
        """Collects loss_stats_dicts from all agents, prepends agent_id to all keys, and joins results."""
        result_dict = {}
        for agent_id, stats in self._agent_id_to_stats.items():
            if agent_id != self.learning_players_id:
                continue
            agent_loss_stats_dict = stats.get_loss_stats_dict()
            for k, v in agent_loss_stats_dict.items():
                result_dict[f"{agent_id}/" + k] = v
        return result_dict

# ======== Step 1: Environment setup =========
train_envs = DummyVectorEnv([_get_env for _ in range(10)])
test_envs = DummyVectorEnv([_get_env for _ in range(10)])

# seed
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
train_envs.seed(seed)
test_envs.seed(seed)

# ======== Step 2: Agent setup =========
policy, optim, agents = _get_agents()

p2_policy = policy.policies['player_2'].model
# freeze(p1_policy)
# print_params(p2_policy)
# print("P:", p1_policy)

# ======== Step 3: Collector setup =========
train_collector = Collector(
    policy,
    train_envs,
    VectorReplayBuffer(20_000, len(train_envs)),
    exploration_noise=True,
)
test_collector = Collector(policy, test_envs, exploration_noise=False)
# policy.set_eps(1)
train_collector.collect(n_step=64 * 10)  # batch size * training_num

# ======== Step 4: Callback functions setup =========
def save_best_fn(policy):
    model_save_path = os.path.join("log", "ttt", "dqn", "policy.pth")
    os.makedirs(os.path.join("log", "ttt", "dqn"), exist_ok=True)
    torch.save(policy.policies[agents[1]].state_dict(), model_save_path)

def stop_fn(mean_rewards):
    return mean_rewards >= 0.9

def train_fn(epoch, env_step):
    policy.policies[agents[1]].set_eps(0.1)
    policy.policies[agents[1]].set_eps(0.1)

def test_fn(epoch, env_step):
    policy.policies[agents[1]].set_eps(0.05)
    policy.policies[agents[1]].set_eps(0.05)

def reward_metric(rews):
    return rews[:, 1]

collect_result = test_collector.collect(n_episode=100)

# print(collect_result)
# print(type(collect_result["rews"].mean(axis=0)))
# print("Average rew is {}".format(collect_result['rew']))
# print(train_collector.buffer)

# ======== Step 5: Run the trainer =========
trainer = OffpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=20,
    step_per_epoch=1000,
    step_per_collect=50,
    episode_per_test=10,
    batch_size=64,
    train_fn=train_fn,
    test_fn=test_fn,
#     stop_fn=stop_fn,
    save_best_fn=save_best_fn,
    update_per_step=0.1,
    test_in_train=False,
    reward_metric=reward_metric,
#     show_progress=False,
)

# print_params(p1_policy)

result = trainer.run()

# return result, policy.policies[agents[1]]
print(f"\n==========Result==========\n{result}")
print("\n(the trained policy can be accessed via policy.policies[agents[1]])")













