from typing import List

import numpy as np
import torch

GAME_NAMES = [
    'AirRaid', 'Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids', 'Atlantis',
    'BankHeist', 'BattleZone', 'BeamRider', 'Berzerk', 'Bowling', 'Boxing',
    'Breakout', 'Carnival', 'Centipede', 'ChopperCommand', 'CrazyClimber',
    'DemonAttack', 'DoubleDunk', 'ElevatorAction', 'Enduro', 'FishingDerby',
    'Freeway', 'Frostbite', 'Gopher', 'Gravitar', 'Hero', 'IceHockey',
    'Jamesbond', 'JourneyEscape', 'Kangaroo', 'Krull', 'KungFuMaster',
    'MontezumaRevenge', 'MsPacman', 'NameThisGame', 'Phoenix', 'Pitfall',
    'Pong', 'Pooyan', 'PrivateEye', 'Qbert', 'Riverraid', 'RoadRunner',
    'Robotank', 'Seaquest', 'Skiing', 'Solaris', 'SpaceInvaders', 'StarGunner',
    'Tennis', 'TimePilot', 'Tutankham', 'UpNDown', 'Venture', 'VideoPinball',
    'WizardOfWor', 'YarsRevenge', 'Zaxxon'
]  # 60
_FULL_ACTION_SET = [
    'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
    'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
    'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
]
ATARI_NUM_ACTIONS = len(_FULL_ACTION_SET)  # 18

_LIMITED_ACTION_SET = {
    'AirRaid': ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'],
    'Alien': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Amidar': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPFIRE', 'RIGHTFIRE',
        'LEFTFIRE', 'DOWNFIRE'
    ],
    'Assault': ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'],
    'Asterix': [
        'NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT',
        'DOWNLEFT'
    ],
    'Asteroids': [
        'NOOP',
        'FIRE',
        'UP',
        'RIGHT',
        'LEFT',
        'DOWN',
        'UPRIGHT',
        'UPLEFT',
        'UPFIRE',
        'RIGHTFIRE',
        'LEFTFIRE',
        'DOWNFIRE',
        'UPRIGHTFIRE',
        'UPLEFTFIRE',
    ],
    'Atlantis': ['NOOP', 'FIRE', 'RIGHTFIRE', 'LEFTFIRE'],
    'BankHeist': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'BattleZone': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'BeamRider': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'UPRIGHT', 'UPLEFT', 'RIGHTFIRE',
        'LEFTFIRE'
    ],
    'Berzerk': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Bowling': ['NOOP', 'FIRE', 'UP', 'DOWN', 'UPFIRE', 'DOWNFIRE'],
    'Boxing': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Breakout': ['NOOP', 'FIRE', 'RIGHT', 'LEFT'],
    'Carnival': ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'],
    'Centipede': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'ChopperCommand': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'CrazyClimber': [
        'NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT',
        'DOWNLEFT'
    ],
    'DemonAttack': ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'],
    'DoubleDunk': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'ElevatorAction': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Enduro': [
        'NOOP', 'FIRE', 'RIGHT', 'LEFT', 'DOWN', 'DOWNRIGHT', 'DOWNLEFT',
        'RIGHTFIRE', 'LEFTFIRE'
    ],
    'FishingDerby': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Freeway': ['NOOP', 'UP', 'DOWN'],
    'Frostbite': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Gopher': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE'
    ],
    'Gravitar': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Hero': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'IceHockey': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Jamesbond': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'JourneyEscape': [
        'NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT',
        'DOWNLEFT', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE',
        'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Kangaroo': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Krull': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'KungFuMaster': [
        'NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'DOWNRIGHT', 'DOWNLEFT',
        'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE',
        'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'MontezumaRevenge': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'MsPacman': [
        'NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT',
        'DOWNLEFT'
    ],
    'NameThisGame': ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'],
    'Phoenix': [
        'NOOP', 'FIRE', 'RIGHT', 'LEFT', 'DOWN', 'RIGHTFIRE', 'LEFTFIRE',
        'DOWNFIRE'
    ],
    'Pitfall': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Pong': ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'],
    'Pooyan': ['NOOP', 'FIRE', 'UP', 'DOWN', 'UPFIRE', 'DOWNFIRE'],
    'PrivateEye': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Qbert': ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN'],
    'Riverraid': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'RoadRunner': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Robotank': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Seaquest': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Skiing': ['NOOP', 'RIGHT', 'LEFT'],
    'Solaris': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'SpaceInvaders': ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'],
    'StarGunner': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Tennis': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'TimePilot': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPFIRE', 'RIGHTFIRE',
        'LEFTFIRE', 'DOWNFIRE'
    ],
    'Tutankham': [
        'NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE'
    ],
    'UpNDown': ['NOOP', 'FIRE', 'UP', 'DOWN', 'UPFIRE', 'DOWNFIRE'],
    'Venture': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'VideoPinball': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPFIRE', 'RIGHTFIRE',
        'LEFTFIRE'
    ],
    'WizardOfWor': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPFIRE', 'RIGHTFIRE',
        'LEFTFIRE', 'DOWNFIRE'
    ],
    'YarsRevenge': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
    'Zaxxon': [
        'NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
        'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
        'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE'
    ],
}

# An array that Converts an action from a game-specific to full action set.
LIMITED_ACTION_TO_FULL_ACTION = {
    game_name: np.array(
        [_FULL_ACTION_SET.index(i) for i in _LIMITED_ACTION_SET[game_name]])
    for game_name in GAME_NAMES
}

# An array that Converts an action from a full action set to a game-specific
# action set (Setting 0=NOOP if no game-specific action exists).
FULL_ACTION_TO_LIMITED_ACTION = {
    game_name: np.array([(_LIMITED_ACTION_SET[game_name].index(i)
                          if i in _LIMITED_ACTION_SET[game_name] else 0)
                         for i in _FULL_ACTION_SET]) for game_name in GAME_NAMES
}

def tokenize_actions(env: str, actions: np.ndarray) -> np.ndarray:
    """Tokenize a list of actions for a given environment."""
    assert env in LIMITED_ACTION_TO_FULL_ACTION
    return LIMITED_ACTION_TO_FULL_ACTION[env][actions]

def batch_tokenize_actions(envs: List[str], actions: np.ndarray) -> np.ndarray:
    """Tokenize a batch of actions for a list of environments."""
    assert actions.ndim == 2
    assert len(envs) == actions.shape[0]
    return np.stack(
        [tokenize_actions(env, act) for env, act in zip(envs, actions)]
    )

def batch_tokenize_envs(envs: List[str]) -> np.ndarray:
    """Tokenize a list of environments."""
    def map_test_env_to_similar_train_env(env):
        if env == 'Robotank':
            return 'Phoenix'
        elif env == 'YarsRevenge':
            return 'ChopperCommand'
        elif env == 'Gravitar':
            return 'TimePilot'
        elif env == 'MsPacman':
            return 'Berzerk'
        else:
            return env

    return np.array([
        GAME_NAMES.index(
            map_test_env_to_similar_train_env(env)
        ) for env in envs
    ])

def init_action_masks():
    action_masks = []

    for env in GAME_NAMES:
        action_mask_for_each_env = []
        for action in range(ATARI_NUM_ACTIONS):
            action_mask_for_each_env.append(
                action in LIMITED_ACTION_TO_FULL_ACTION[env]
            )
        action_masks.append(action_mask_for_each_env)
        
    return torch.tensor(action_masks, dtype=torch.bool)

action_masks = init_action_masks()


if __name__ == "__main__":
    print(tokenize_actions("Assault", np.array([0, 1, 2, 3, 4, 5, 6])))
    print(batch_tokenize_envs(["Assault"]))
    print(action_masks)