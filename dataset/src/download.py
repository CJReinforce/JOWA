import itertools
import os
from multiprocessing import Pool
from subprocess import Popen

URI = 'gs://atari-replay-datasets/dqn/{}/{}/replay_logs/'
BASE_DIR = 'dataset/original/'
# 60
# ENVS = [
#     'AirRaid', 'Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids', 'Atlantis',
#     'BankHeist', 'BattleZone', 'BeamRider', 'Berzerk', 'Bowling', 'Boxing',
#     'Breakout', 'Carnival', 'Centipede', 'ChopperCommand', 'CrazyClimber',
#     'DemonAttack', 'DoubleDunk', 'ElevatorAction', 'Enduro', 'FishingDerby',
#     'Freeway', 'Frostbite', 'Gopher', 'Gravitar', 'Hero', 'IceHockey',
#     'Jamesbond', 'JourneyEscape', 'Kangaroo', 'Krull', 'KungFuMaster',
#     'MontezumaRevenge', 'MsPacman', 'NameThisGame', 'Phoenix', 'Pitfall',
#     'Pong', 'Pooyan', 'PrivateEye', 'Qbert', 'Riverraid', 'RoadRunner',
#     'Robotank', 'Seaquest', 'Skiing', 'Solaris', 'SpaceInvaders', 'StarGunner',
#     'Tennis', 'TimePilot', 'Tutankham', 'UpNDown', 'Venture', 'VideoPinball',
#     'WizardOfWor', 'YarsRevenge', 'Zaxxon'
# ]
# 20
ENVS = [
    'assault', 'atlantis', 'beam-rider',
    'berzerk', 'carnival', 'centipede',
    'chopper-command', 'demon-attack',
    'gravitar', 'ms-pacman', 'name-this-game', 
    'phoenix', 'pong', 'robotank', 'seaquest',
    'space-invaders', 'star-gunner', 'time-pilot',
    'yars-revenge', 'zaxxon'
]
# 5
TEST_ENVS = [
    'pong', 'ms-pacman', 'gravitar', 
    'yars-revenge', 'robotank'
]
# 15
TRAIN_ENVS = [i for i in ENVS if i not in TEST_ENVS]


def get_dir_path(env, index, epoch, base_dir=BASE_DIR):
    return os.path.join(base_dir, env, str(index), 'replay_logs')
    
def inspect_dir_path(env, index, epoch, base_dir=BASE_DIR):
    path = get_dir_path(env, index, epoch, base_dir)
    if not os.path.exists(path):
        return False
    for name in ['observation', 'action', 'reward', 'terminal']:
        if not os.path.exists(os.path.join(path, name + '.gz')):
            return False
    return True

def capitalize_game_name(game):
    game = game.replace('-', '_')
    return ''.join([g.capitalize() for g in game.split('_')])

def _download(name, env, index, epoch, dir_path):
    file_name = '$store$_{}_ckpt.{}.gz'.format(name, epoch)
    uri = URI.format(env, index) + file_name
    path = os.path.join(dir_path, file_name)
    p = Popen(['gsutil', '-m', 'cp', '-R', '-n', uri, path])
    p.wait()
    return path

def download_dataset(env, index, epoch, base_dir=BASE_DIR):
    dir_path = get_dir_path(env, index, epoch, base_dir)
    _download('observation', env, index, epoch, dir_path)
    _download('action', env, index, epoch, dir_path)
    _download('reward', env, index, epoch, dir_path)
    _download('terminal', env, index, epoch, dir_path)
    
def main(args):
    env_, index, epoch = args
    env = capitalize_game_name(env_) if env_[0].islower() else env_
    path = get_dir_path(env, index, epoch)
    if not inspect_dir_path(env, index, epoch):
        os.makedirs(path, exist_ok=True)
        download_dataset(env, index, epoch)


if __name__ == "__main__":
    num_processes = 64

    indices = range(1, 6)  # 5 agents
    epochs = range(1, 51)  # 50 epochs

    all_args = list(itertools.product(ENVS, indices, epochs))

    with Pool(num_processes) as pool:
        pool.map(main, all_args)
