import argparse
import ast
import os
import re
from pprint import pprint

import pandas as pd
import scipy


# if .log ends unexpectedly
clip_undone = True
clip_least_timestep = 0

original_returns_line_from_last = 5
original_clipped_returns_line_from_last = 11


def read_last_line(file_path, last_line: int = 5):
    with open(file_path, 'rb') as file:
        file.seek(0, 2)
        position = file.tell()
        total = position
        
        lines_to_read = last_line + 1
        current_line = 0
        
        while position >= 0 and current_line < lines_to_read:
            file.seek(position)
            char = file.read(1)
            if char == b'\n' and position != total - 1:  # 检测换行符
                current_line += 1
            position -= 1
        
        if current_line == lines_to_read:
            file.readline()
            sixth_last_line = file.readline().decode().strip()
            return sixth_last_line
        else:
            return ''


def collect_scores(files):
    hyper_params_pattern = r'buffer_size_(\d+?)_plan_(.+?)_bw_(\d)_h_(\d)'
    all_data = {}

    for file in files:
        original_returns = read_last_line(file, original_returns_line_from_last)
        original_clipped_returns = read_last_line(file, original_clipped_returns_line_from_last)
        if not original_returns.startswith('Original returns: '):
            print(f'Warning: {file} has not find the original returns line.')
            if clip_undone:
                for i in range(1, 20):
                    original_returns = read_last_line(file, i)
                    temp = re.search(r"timestep': (\d+).+return': (\d+\.?\d*),", original_returns)
                    if temp is not None:
                        timestep = int(temp.group(1))
                        if timestep >= clip_least_timestep:
                            original_returns = [float(temp.group(2))]
                            break
                        else:
                            continue
                    else:
                        continue
            else:
                continue
        else:
            original_returns = ast.literal_eval(original_returns[18:])
            
        if not original_clipped_returns.startswith('Original clipped_returns:'):
            print(f'Warning: {file} has not find the original clipped returns line.')
            if clip_undone:
                for j in range(i, 40):
                    original_clipped_returns = read_last_line(file, j)
                    temp = re.search(r"timestep': (\d+).+clipped_return': (\d+\.?\d*),", original_clipped_returns)
                    if temp is not None:
                        timestep = int(temp.group(1))
                        if timestep >= clip_least_timestep:
                            original_clipped_returns = [float(temp.group(2))]
                            break
                        else:
                            continue
                    else:
                        continue
            else:
                continue
        else:
            original_clipped_returns = ast.literal_eval(original_clipped_returns[26:])

        if type(original_returns) == str:
            continue
        
        if type(original_clipped_returns) == str:
            continue
        
        hyper_params = re.search(hyper_params_pattern, file)
        ngs, use_plan, bw, h = hyper_params.group(1,2,3,4)
        
        if all_data.get((ngs, use_plan, bw, h, 'R')) is None:
            all_data[(ngs, use_plan, bw, h, 'R')] = original_returns
            all_data[(ngs, use_plan, bw, h, 'CR')] = original_clipped_returns
        else:
            all_data[(ngs, use_plan, bw, h, 'R')].extend(original_returns)
            all_data[(ngs, use_plan, bw, h, 'CR')].extend(original_clipped_returns)
    
    return all_data
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--name', '--model-name', type=str, default='JOWA_150M')
    parser.add_argument('-E', '--env', type=str)
    args = parser.parse_args()

    model_name, env = args.name, args.env

    logs_pattern = r'.*?{}.*?_play_{}.*?\.log'.format(model_name, env)
    files = [i for i in os.listdir('./') if re.fullmatch(logs_pattern, i)]

    pprint(files)
    temp = input(f'There are {len(files)} files. Is that right?')

    if temp.lower() in ['y', 'yes']:
        all_data = collect_scores(files)

        df = pd.DataFrame.from_dict(all_data, orient='index')
        df['Mean'] = df.values.mean(1)
        df['IQM'] = scipy.stats.trim_mean(df.values, proportiontocut=0.25, axis=1)
        pprint(df)
        df.to_csv(f'{model_name}_play_{env}_aggregaded_scores.csv')

        temp = input('Need to delete logs?')
        if temp.lower() in ['y', 'yes']:
            for file in files:
                os.remove(file)
            print('Delete selected files.')