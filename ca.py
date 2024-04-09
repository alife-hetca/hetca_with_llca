########################### GOL WITH CELL THREE STATES: ALIVE, DECAY AND QUIESCENT      #####################

import copy
from utils import animation
from utils import globals
import life_like_ca as ca
import os
from datetime import datetime
import numpy as np
import argparse
import random
from utils import analyse_results as analyse
from os import system
import importlib

RESULT_DIR = './animation'


def generate_energy(lowest_cell_energy: int, highest_energy_level: int, cell_rules: np.ndarray):
    values = np.arange(lowest_cell_energy, highest_energy_level + 1)
    # divide the number between 1 to 0.01
    probability_distribution = np.linspace(1, 0.01, len(values))

    # Normalize the probabilities -- total probability 1
    probability_distribution /= probability_distribution.sum()
    new_energy = np.random.choice(
        values, size=cell_rules.shape, p=probability_distribution)
    new_energy[cell_rules == 'nan'] = 0
    return new_energy


def create_needed_folders(param: dict):

    # main directory
    result_path = param['result_dir']
    amax_val = param['amax']
    adec_val = param['adec']
    mut_val = param['mutation_rate']
    prob_to_get_neigh_genome_val = param['probability_to_get_neighbour_genome']
    grid_size = param['cell_states'].shape[0] if 'cell_states' in param else param['grid_size']
    iteration = param['iteration']

    grid = f'grid_{grid_size}*{grid_size}'
    amax = f'amax_{amax_val}'
    adec = f'adec_{adec_val}'
    mut = f'mut_{mut_val}'
    prob_to_get_neigh_genome = f'prob_to_get_neigh_genome_{prob_to_get_neigh_genome_val}'
    iter = f'iter_{iteration}'

    now = datetime.now()
    current_year = str(now.year)
    current_month = str(now.month)
    current_day = str(now.day)
    date_dir = f'{result_path}/{current_year}_{current_month}_{current_day}'
    current_hour = str(now.hour)
    current_minute = str(now.minute)
    current_second = str(now.second)
    current_milli_second = str(now.microsecond)
    cur_time = f'{current_hour}:{current_minute}:{current_second}:{current_milli_second}'

    ca_output_dir = f'{date_dir}/{grid}__{iter}__{amax}__{adec}__{mut}__{prob_to_get_neigh_genome}__{cur_time}'
    metadata_dir = f'{ca_output_dir}/metadata'
    animation_dir = f'{ca_output_dir}/animations'
    plot_dir = f'{ca_output_dir}/plots'

    os.makedirs(result_path, exist_ok=True)
    os.makedirs(date_dir, exist_ok=True)
    os.makedirs(ca_output_dir, exist_ok=True)
    param['exp_result_dir'] = ca_output_dir
    os.makedirs(metadata_dir, exist_ok=True)
    param['metadata_dir'] = metadata_dir
    os.makedirs(animation_dir, exist_ok=True)
    param['animation_dir'] = animation_dir
    os.makedirs(plot_dir, exist_ok=True)
    param['plot_dir'] = plot_dir


def run_ca(param):
    cell_states, cell_ages, cell_rules, grid_states, cell_energy = [], [], [], [], []

    iteration = param['iteration']
    llca = ca.LLCA(param)
    energy_distribution_times = int(
        param['energy_distribution_ratio'] * iteration)
    iteration_to_introduce_energy = sorted(
        random.sample(range(iteration), energy_distribution_times))
    print("Processing for ", iteration, " iteration")
    print('iteration_to_introduce_energy : ', iteration_to_introduce_energy)
    for i in range(iteration):
        if i > 0:
            llca.iterate()

        if i in iteration_to_introduce_energy:
            print('energy iteration : ', i)
            new_energy = generate_energy(
                param['lowest_energy_level'], param['highest_energy_level'], llca.cell_rules)
            llca.energy += copy.deepcopy(new_energy)
            # energy_df.loc[i] = [new_energy, copy.deepcopy(llca.energy)]

        cell_states.append(copy.deepcopy(llca.cell_states))
        cell_ages.append(copy.deepcopy(llca.cell_ages))
        cell_rules.append(copy.deepcopy(llca.cell_rules))
        grid_states.append(copy.deepcopy(llca.grid_state))
        cell_energy.append(copy.deepcopy(llca.energy))
        print("Iteration  ", i, " Completed.")

    return {'cell_states': cell_states,
            'cell_ages': cell_ages,
            'cell_rules': cell_rules,
            'grid_states': grid_states,
            'cell_energy': cell_energy}


def set_default_parameters(other_params=None):
    param = {}
    param['grid_size'] = globals.GRID_SIZE
    param['iteration'] = globals.ITERATION
    param['amax'] = globals.AMAX
    param['adec'] = globals.ADEC
    param['mutation_rate'] = globals.MUTATION_RATE

    # default_initial_cell_state is set to False if no initial cell states are passed,
    # otherwise it needs to be set True along with 'cell states'
    param['default_initial_cell_state'] = globals.DEFAULT_INITIAL_CELL_STATES
    if param['default_initial_cell_state']:

        cell_states = globals.DEFAULT_CELL_STATES
        grid_states = globals.DEFAULT_GRID_STATES
        q_check = len(np.where((cell_states == 'q') & (grid_states == 1))[0])

        if (cell_states.shape != grid_states.shape) or q_check > 0:
            print('Cell states properties do not align with Grid states properties')
            print('Exiting...')
            exit()

        param['cell_states'] = cell_states
        param['grid_states'] = grid_states
    else:
        grid_size = (globals.GRID_SIZE, globals.GRID_SIZE)
        probability = [globals.INITIAL_ALIVE_CELL_STATE_RATIO,
                       1 - globals.INITIAL_ALIVE_CELL_STATE_RATIO]
        cell_states = np.random.choice(
            ['a', 'q'], size=grid_size, p=probability)

        grid_states = np.zeros(cell_states.shape, dtype=int)
        a_positions = cell_states == 'a'
        grid_prob = [1 - globals.ALIVE_CELL_INTIAL_1_VALUE_PROB,
                     globals.ALIVE_CELL_INTIAL_1_VALUE_PROB]
        grid_states[a_positions] = np.random.choice(
            [0, 1], size=a_positions.sum(), p=grid_prob)
        # making double sure, not needed at all.
        grid_states[cell_states == 'q'] = 0

        param['cell_states'] = cell_states
        param['grid_states'] = grid_states

    param['probability_to_get_neighbour_genome'] = globals.PROB_TO_GET_ELIGIBLE_NEIGHBOURS
    param['result_dir'] = RESULT_DIR
    param['grid_size'] = globals.GRID_SIZE
    param['initial_rule'] = globals.INITIAL_RULE
    param['eligible_cell_states'] = globals.ELIGIBLE_CELL_STATES

    # Energy configurations
    param['energy_depletion'] = globals.ENERGY_DEPLETION
    param['energy_distribution_ratio'] = globals.ENERGY_DISTRIBUTION_RATIO
    param['lowest_energy_level'] = globals.LOWEST_ENERGY_LEVEL
    param['highest_energy_level'] = globals.HIGHEST_ENERGY_LEVEL

    #File names
    param['cellstate_filename'] = globals.ANIM_CELL_STATE
    param['gridstate_filename'] = globals.ANIM_GRID_STATE
    param['cellrule_filename'] = globals.RULES_ANIM_FILE_NAME

    if other_params:
        param = param | other_params

    return param


def copy_file(source, destination):
    with open(source, 'r') as source_file:
        source_code = source_file.read()
    with open(destination, 'w') as destination:
        destination.write(source_code)


def copy_params_to_utils(global_path, default_param_path='utils/globals.py'):
    global RESULT_DIR
    if (global_path != default_param_path):
        copy_file(global_path, default_param_path)
        file_name = global_path.rfind('/')
        RESULT_DIR = global_path[:file_name]


def start_process(global_location):
    start_datetime = datetime.now()
    print('STARTED APPLICATION AT: ', start_datetime)

    default_param_path = 'utils/globals.py'
    copy_params_to_utils(global_location, default_param_path)
    importlib.reload(globals)
    # Set default parameters
    params = set_default_parameters()
    create_needed_folders(params)
    print(params)

    # Run CA
    ca_properties = run_ca(params)

    # Save numpy values of properties
    # np.savez(f'{params["metadata_dir"]}/cell_states.npz', *ca_properties['cell_states'])
    # np.savez(f'{params["metadata_dir"]}/cell_ages.npz', *ca_properties['cell_ages'])
    # np.savez(f'{params["metadata_dir"]}/cell_rules.npz', *ca_properties['cell_rules'])
    # np.savez(f'{params["metadata_dir"]}/grid_states.npz', *ca_properties['grid_states'])
    # np.savez(f'{params["metadata_dir"]}/cell_energy.npz', *ca_properties['cell_energy'])

    # Save parameters
    copy_file(default_param_path, f'{params["metadata_dir"]}/parameters.py')

    # Create and Save Plots and Animations
    analysed_results = analyse.process_data_needed_for_plot(
        ca_properties, params['plot_dir'])
    analysed_results['iteration'] = params['iteration']



    # directories = animation.visual_result(ca_properties, params)

    end_datetime = datetime.now()
    print('ENDED APPLICATION AT: ', end_datetime)
    print('Total Time Taken',  end_datetime - start_datetime)
    print("COMPLETE")
    return params['exp_result_dir'], analysed_results


if __name__ == "__main__":
    default_param_path = 'utils/globals.py'
    parser = argparse.ArgumentParser()
    parser.add_argument('global_path', nargs='?', type=str, default=default_param_path,
                        help='Path of globals where there are all parameters')
    global_path = parser.parse_args().global_path
    start_process(global_path)
