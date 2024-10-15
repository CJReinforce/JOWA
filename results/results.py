import json
import os
import scipy

import numpy as np
import pandas as pd

from base_scores import *

data = json.load(open('results/data.json', 'r'))
table = []

for algo, scores in data.items():
    subtable = {
        'Raw Score': [],
        'HNS': [],
        'DNS': []
    }
    indexes = []
    superhuman = 0

    for key, mean_score in scores.items():
        indexes.append(key)
        try:
            random, human, dqn = lookup_score_for_env(key)
        except:
            import ipdb; ipdb.set_trace()
        human_normal_score = normalize(random, human, mean_score)
        dqn_normal_score = normalize(random, dqn, mean_score)

        subtable['Raw Score'].append(mean_score)
        subtable['HNS'].append(human_normal_score)
        subtable['DNS'].append(dqn_normal_score)

        if mean_score > human:
            superhuman += 1
    
    num_games = len(indexes)
    np_HNS = np.array(subtable['HNS'][:num_games])
    np_DNS = np.array(subtable['DNS'][:num_games])

    indexes.append('Superhuman')
    subtable['HNS'].append(superhuman)
    subtable['DNS'].append(superhuman)

    indexes.append('Median NS')
    subtable['HNS'].append(np.median(np_HNS))
    subtable['DNS'].append(np.median(np_DNS))

    indexes.append('IQM NS')
    subtable['HNS'].append(
        scipy.stats.trim_mean(np_HNS, proportiontocut=0.25, axis=None)
    )
    subtable['DNS'].append(
        scipy.stats.trim_mean(np_DNS, proportiontocut=0.25, axis=None)
    )

    dummy_values = [None, None, None]
    subtable['Raw Score'].extend(dummy_values)

    df = pd.DataFrame(subtable, index=indexes)
    last_rows = df.iloc[-len(dummy_values):]
    df_to_sort = df.iloc[:-len(dummy_values)]
    df_sorted = df_to_sort.sort_index()
    df = pd.concat([df_sorted, last_rows])
    
    df.columns = pd.MultiIndex.from_product([[algo], df.columns])
    table.append(df)

table = pd.concat(table, axis=1)
with pd.option_context('display.float_format', '{:.3f}'.format):
    print(table)
table.to_csv('results/results.csv')