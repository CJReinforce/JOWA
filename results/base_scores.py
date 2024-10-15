ALL_GAMES = [
    'Alien',
    'Amidar',
    'Assault',
    'Asterix',
    'Atlantis',
    'BankHeist',
    'BattleZone',
    'BeamRider',
    'Berzerk',
    'Boxing',
    'Breakout',
    'Carnival',  
    'Centipede',
    'ChopperCommand',
    'CrazyClimber',
    'DemonAttack',
    'DoubleDunk',
    'Enduro',
    'FishingDerby',
    'Freeway',
    'Frostbite',
    'Gopher',
    'Gravitar',
    'Hero',
    'IceHockey',
    'Jamesbond',
    'Kangaroo',
    'Krull',
    'KungFuMaster',
    'MsPacman',
    'NameThisGame',
    'Phoenix',
    'Pong',
    'Pooyan',  
    'Qbert',
    'Riverraid',
    'Robotank',
    'Seaquest',
    'SpaceInvaders',
    'StarGunner',
    'TimePilot',
    'UpNDown',
    'VideoPinball',
    'WizardOfWor',
    'YarsRevenge',
    'Zaxxon'
]

RANDOM_PERFORMANCE = [
    227.8,
    5.8,
    222.4,
    210.0,
    12_850.0,
    14.2,
    2_360.0,
    363.9,
    123.7,
    0.1,
    1.7,
    700.8,
    2_090.9,
    811.0,
    10_780.5,
    152.1,
    -18.6,
    0.0,
    -91.7,
    0.0,
    65.2,
    257.6,
    173.0,
    1_027.0,
    -11.2,
    29.0,
    52.0,
    1_598.0,
    258.5,
    307.3,
    2_292.3,
    761.4,
    -20.7,
    371.2,
    163.9,
    1_338.5,
    2.2,
    68.4,
    148.0,
    664.0,
    3_568.0,
    533.4,
    16_256.9,
    563.5,
    3_092.9,
    32.5,
]

HUMAN_PERFORMANCE = [
    7_127.7,
    1_719.5,
    742.0,
    8_503.3,
    29_028.1,
    753.1,
    37_187.5,
    16_926.5,
    2630.4,
    12.1,
    30.5,
    3800,  # Carnival
    12_017.0,
    7_387.8,
    35_829.4,
    1_971.0,
    -16.4,
    860.5,
    -38.7,
    29.6,
    4_334.7,
    2_412.5,
    3_351.4,
    30_826.4,
    0.9,
    302.8,
    3_035.0,
    2_665.5,
    22_736.3,
    6_951.6,
    8_049.0,
    7_242.6,
    14.6,
    0,  # Pooyan
    13_455.0,
    17_118.0,
    11.9,
    42_054.7,
    1_668.7,
    10_250.0,
    5_229.2,
    11_693.2,
    17_667.9,
    4_756.5,
    54_576.9,
    9_173.3,
]

DQN_PERFORMANCE = [
    2484.5,
    1207.7,
    1525.2,
    2711.4,
    853640.0,
    601.8,
    17784.8,
    5852.4,
    487.5,
    78.0,
    96.2,
    4784.8,
    2583.0,
    2690.6,
    104568.8,
    6361.6,
    -6.5,
    628.9,
    0.6,
    26.3,
    367.1,
    5479.9,
    330.1,
    17325.4,
    -5.8,
    573.3,
    11486.0,
    6097.6,
    23435.4,
    3402.4,
    7278.6,
    4996.6,
    16.6,
    3212.0,
    10117.5,
    11638.9,
    59.8,
    1600.7,
    1794.2,
    42165.2,
    3654.4,
    8488.3,
    63406.1,
    2065.8,
    23909.4,
    4538.6
]

def capitalize_game_name(game: str):
    game = game.replace('-', '_')
    return ''.join([g.capitalize() for g in game.split('_')])

def lookup_score_for_env(game: str):
    if game[0].islower():
        game = capitalize_game_name(game)

    index = ALL_GAMES.index(game)
    return RANDOM_PERFORMANCE[index], HUMAN_PERFORMANCE[index], DQN_PERFORMANCE[index]

def normalize(_min, _max, ori_score):
    min_score = min(_min, _max)
    max_score = max(_min, _max)
    norm_score = (ori_score - min_score) / (max_score - min_score)
    return norm_score


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--game', type=str)
    parser.add_argument('-s', '--score', type=float)
    args = parser.parse_args()
    random, human, dqn = lookup_score_for_env(args.game)

    if args.score is not None:
        print(f"human normalized score: {normalize(random, human, args.score)}")
        print(f"DQN normalized score: {normalize(random, dqn, args.score)}")
    else:
        print(f"random score: {random}, human score: {human}")