import ca
import yaml
from utils import analyse_results as analysis
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import argparse



config_path = './config/config.yaml'


def get_max_for_y_limit(exp_results):
    max_cumulative_rules, max_grid_difference, max_grid_state_count, max_cell_state_count = 0, 0, 0, 0

    for result in exp_results:
        
        max_cell_state_count = max(max_cell_state_count, max (result['alive_cell_count'] + result['decay_cell_count'] + result['dead_cell_count']))
        max_cumulative_rules = max(max_cumulative_rules,
                                   max(result['cumulative_rule_count']))
        max_grid_state_count = max(max_grid_state_count,
                                   max(
                                       result['count_alive_1'] +
                                       result['count_alive_0'] +
                                       result['dead_cell_count']))
        max_grid_difference = max(
            max_grid_difference, max(result['grid_difference']))

    return {
        'max_cumulative_rules': max_cumulative_rules,
        'max_grid_difference': max_grid_difference,
        'max_grid_state_count': max_grid_state_count,
        'max_cell_state_count': max_cell_state_count
    }


def plot_individual_plot_of_experiment(exp_results: dict, max_dict: dict):
    for dir, result in exp_results.items():
        plot_dir = f'{dir}/plots'
        analysis.plot_cell_states_count(
            result['alive_cell_count'], result['decay_cell_count'], result['dead_cell_count'], max_dict['max_cell_state_count'], plot_dir)
        analysis.plot_cumm_unique_rules_count(
            result['cumulative_rule_count'], max_dict['max_cumulative_rules'], plot_dir)
        analysis.plot_count_1s_0s_dead_in_grid_state(
            result['count_alive_0'], result['count_alive_1'], result['dead_cell_count'], max_dict['max_grid_state_count'], plot_dir)
        analysis.plot_difference_betn_two_grid_states(
            result['grid_difference'], max_dict['max_grid_difference'], plot_dir)


def visualize_set_of_exp(all_exp_data: list, ylimit_dict: dict):
    for exp_set in all_exp_data:
        key = list(exp_set.keys())[0]
        split_key = key.split('/')
        exp_name = split_key[-3]
        plot_path = '/'.join(split_key[:-2]) + '/' + exp_name
        os.makedirs(plot_path, exist_ok=True)

        experiment_results = list(exp_set.values())
        unique_iter = list(set([d["iteration"] for d in experiment_results]))
        if len(unique_iter) > 1:
            print("Number of iteration is different in exp: ", exp_name)
            break

        iter = int(unique_iter[0])
        # iter = 500
        print(exp_name)

        analysis.visualize_each_exp_set(experiment_results,
                                        exp_name,
                                        iter,
                                        ylimit_dict,
                                        plot_path
                                        )


def begin_experimentation(file_locations, no_of_experiments):
    all_exp_data = []

    for g in file_locations:
        exp_results = {}
        for e in range(no_of_experiments):
            dir, exp_result = ca.start_process(g)
            exp_results[dir] = exp_result
        all_exp_data.append(exp_results)

    merged_exp_dict = dict(
        **{k: v for d in all_exp_data for k, v in d.items()})
    max_dict = get_max_for_y_limit(list(merged_exp_dict.values()))
    plot_individual_plot_of_experiment(merged_exp_dict, max_dict)
    visualize_set_of_exp(all_exp_data, max_dict)


if __name__ == "__main__":
    start_time = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument('global_path', nargs='?', type=str, default=config_path,
                        help='Path of globals where there are all parameters')
    config_file = parser.parse_args().global_path

    with open(config_file, 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)
    file_locations = data['file_locations']
    no_of_experiments = data['no_of_experiments']
    begin_experimentation(file_locations, no_of_experiments)
    print('TOTAL TIME TAKEN: ', datetime.now() - start_time)
