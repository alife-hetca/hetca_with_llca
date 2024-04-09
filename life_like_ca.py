
'''
The given code defines a class LLCA which represents a Life-Like Cellular Automaton (LLCA) 
with three cell states: alive, decay, and quiescent.

'''

import numpy as np
import random
import copy
# import pandas as pd

class LLCA:

    '''
        Initializes the LLCA object. It sets the initial grid_state, cell rules, cell ages, 
        and cell states based on the specified rule_type. 
        It also initializes other parameters and variables used in the simulation.

        param is a dictionary where we can pass values as desired.
        The default value for param is None, which means it can be omitted when calling the function.
        If omitted, it will use a value from the global scope.
    '''

    def __init__(self, param):

        self.cell_states = param['cell_states']
        self.grid_state = param['grid_states']

        # set initial cell ages and cell rules
        self.cell_ages = np.where(self.cell_states == 'a', 1, 0)
        self.cell_rules = np.where(self.cell_states == 'q', np.nan, param['initial_rule'])

        # set energy
        self.energy = np.zeros(self.cell_states.shape, dtype=int)

        self.amax = param['amax']
        self.adec = param['adec']
        self.mutation_rate = param['mutation_rate']
        self.probability_to_get_neighbour_genome = param['probability_to_get_neighbour_genome']
        self.directory = param['result_dir']
        self.eligible_cell_states = param['eligible_cell_states']
        self.energy_depletion = param['energy_depletion']
        
   
    '''
        Parses a rule string into birth and survival conditions. 
        It returns two arrays representing the birth (B) and survival (S) conditions.
    '''

    def parse_rule(self, rule):
        r = rule.upper().split("S")
        B = np.array([int(i) for i in r[0][1:]]).astype(np.int64)
        S = np.array([]).astype(np.int64) if (r[1] == '') else np.array(
            [int(i) for i in r[1]]).astype(np.int64)
        return B, S

    '''
        Computes the number of living neighbors for each cell in the LLCA grid. 
        It returns a matrix representing the neighbor counts.
    '''

    def neighbors(self):
        grid_state = self.grid_state
        N = np.zeros(grid_state.shape, dtype=np.int8)  # Neighbors matrix
        N[:-1, :] += grid_state[1:, :]  # Living cells south
        N[:, :-1] += grid_state[:, 1:]  # Living cells east
        N[1:, :] += grid_state[:-1, :]  # Living cells north
        N[:, 1:] += grid_state[:, :-1]  # Living cells west
        N[:-1, :-1] += grid_state[1:, 1:]  # Living cells south east
        N[1:, :-1] += grid_state[:-1, 1:]  # Living cells north east
        N[1:, 1:] += grid_state[:-1, :-1]  # Living cells north west
        N[:-1, 1:] += grid_state[1:, :-1]  # Living cells south west
        return N

    '''
        Applies mutations to a given rule. 
        It randomly mutates the birth and survival conditions based on the specified mutation rates. 
        It returns the mutated rule as a string.
    '''

    def mutate_rule(self, rule: str):
        choice = random.choice(["BORN", "SURVIVE"])
        born_rule, survival_rule = self.parse_rule(rule)
        if choice == "BORN":
            born_rule = self.apply_mutation(born_rule)
        else:
            survival_rule = self.apply_mutation(survival_rule)

        new_rule = "B" + str(''.join(np.array2string(born_rule)[1:-1].split()))+"S"+str(
            ''.join(np.array2string(survival_rule)[1:-1].split()))

        return new_rule

    '''
        Applies a rule to a cell based on its current state, the number of living neighbors, and the current cell value. 
        It returns the next state of the cell 
    '''

    def apply_rule(self, rule: str, no_of_neighs: int, grid_val: int):
        B, S = self.parse_rule(rule)
        return 1 if ((grid_val == 0) & (np.isin(no_of_neighs, B))) or (
            (grid_val == 1) & (np.isin(no_of_neighs, S))) else 0

    '''
        Performs one iteration of the LLCA simulation. 
        It updates the grid_state, cell rules, cell ages, and cell states based on the current state and the applied rules.
    '''

    def iterate(self):
        neighbours = self.neighbors()
        grid_state = self.grid_state
        cell_rules = self.cell_rules
        next_cell_ages = copy.deepcopy(self.cell_ages)
        next_cell_states = copy.deepcopy(self.cell_states)
        next_cell_rules = copy.deepcopy(self.cell_rules)
        next_cell_energy = copy.deepcopy(self.energy)
        next_grid_state = np.zeros(grid_state.shape, dtype=np.int8)

        for i in range(grid_state.shape[0]):
            for j in range(grid_state.shape[0]):
                state = self.cell_states[i][j]
                if state == 'q':
                    eligible_neighbours = self.get_eligible_neighbours(i, j)
                    if len(eligible_neighbours) > 0:
                        select_eligible_neighbour = round(
                            random.uniform(0, 1), 1)
                        
                        if select_eligible_neighbour <= self.probability_to_get_neighbour_genome:
                            # select eligible neighbour
                            neigh_index = random.choice(eligible_neighbours)
                            # Transfer rules
                            next_cell_rules[i][j] = self.cell_rules[neigh_index[0]
                                                                    ][neigh_index[1]]

                            mutation_choice = round(
                                random.uniform(0, 1), 1)  # mutate rule
                            if mutation_choice != 0 and mutation_choice <= self.mutation_rate:
                                a = self.mutate_rule(next_cell_rules[i][j])
                                next_cell_rules[i][j] = a

                            # Change cell state to living
                            next_cell_states[i][j] = 'a'
                            next_cell_ages[i][j] = 1  # reset age to 0
                            next_cell_energy[i][j] = 0



                elif (state == 'd'):
                    if next_cell_energy[i][j] > 0:
                        next_cell_energy[i][j] -= self.energy_depletion
                    else:
                        next_cell_ages[i][j] += 1
                    next_grid_state[i][j] = grid_state[i][j]
                    if (next_cell_ages[i][j] > self.adec):
                        # if cell states goes to 'q' it loses all of its information including energy
                        next_cell_states[i][j] = 'q'
                        next_grid_state[i][j] = 0
                        next_cell_ages[i][j] = 0
                        next_cell_rules[i][j] = np.nan
                        next_cell_energy[i][j] = 0

                else:
                    if next_cell_energy[i][j] > 0:
                        next_cell_energy[i][j] -= self.energy_depletion
                    else:
                        next_cell_ages[i][j] += 1

                    # apply rule
                    next_grid_state[i][j] = self.apply_rule(
                        cell_rules[i][j], neighbours[i][j], grid_state[i][j])
                    if next_cell_ages[i][j] > self.amax:
                        next_cell_states[i][j] = 'd'

        self.grid_state[:] = copy.deepcopy(next_grid_state)
        self.cell_rules[:] = copy.deepcopy(next_cell_rules)
        self.cell_states[:] = copy.deepcopy(next_cell_states)
        self.cell_ages[:] = copy.deepcopy(next_cell_ages)
        self.energy[:] = copy.deepcopy(next_cell_energy)

    '''
        Applies a mutation to either the birth or survival conditions of a rule. 
        It randomly adds, removes or substitute a digit from the condition. 
        It returns the mutated condition as an array.
    '''

    def apply_mutation(self, rule: np.ndarray):
        rule_str = ''.join(str(i) for i in rule)
        # choice decides whether to add new rule, remove rule or replace.
        choice = random.choice(["ADD", "REMOVE", "REPLACE"])
        if choice == 'ADD':
            if (len(rule) < 9):
                
                # get those digits that are not in already existing rule
                eligible_digits = [x for x in range(9) if x not in rule]
                # select random rule from the eligible_digits
                random_number = random.choice(eligible_digits)
                # concat to existing rule.
                rule_str = rule_str + str(random_number)
        elif choice == 'REMOVE':  # remove one rule
            if (len(rule_str) > 0):
                random_index = random.randint(0, len(rule_str)-1)
                rule_str = rule_str[:random_index] + rule_str[random_index+1:]
        else:
            if 0 < len(rule_str) < 9:
                random_index = random.randint(0, len(rule_str)-1)
                # get those digits that are not in already existing rule
                eligible_digits = [x for x in range(9) if x not in rule]
                random_number = random.choice(eligible_digits)
                rule_str = rule_str[:random_index] + \
                    str(random_number) + rule_str[random_index+1:]

        rule = [int(digit) for digit in rule_str]
        final_rule = np.array(rule)
        final_rule.sort()
        return final_rule

    def index_exists(self, row, col):
        return True if (row >= 0 and row < self.grid_state.shape[0]) and (col >= 0 and col < self.grid_state.shape[1]) else False

    '''
        Retrieves the coordinates of eligible neighboring cells that are currently alive. 
        It returns a list of coordinate tuples.
    '''

    def get_eligible_neighbours(self, r: int, c: int):
        eligible_neighbours = []
        eligible_cell_state = self.eligible_cell_states
        if (self.index_exists(r, c - 1)) and (self.cell_states[r][c - 1] in eligible_cell_state):
            eligible_neighbours.append((r, c - 1))  # get neighbour in West
        if (self.index_exists(r - 1, c)) and (self.cell_states[r - 1][c] in eligible_cell_state):
            eligible_neighbours.append((r - 1, c))  # get neighbour in North
        if (self.index_exists(r, c + 1)) and (self.cell_states[r][c + 1] in eligible_cell_state):
            eligible_neighbours.append((r, c + 1))  # get neighbour in East
        if (self.index_exists(r + 1, c)) and (self.cell_states[r + 1][c] in eligible_cell_state):
            eligible_neighbours.append((r + 1, c))  # get neighbour in South
        if (self.index_exists(r - 1, c - 1)) and (self.cell_states[r - 1][c - 1] in eligible_cell_state):
            # get neighbour in NorthWest
            eligible_neighbours.append((r - 1, c - 1))
        if (self.index_exists(r - 1, c + 1)) and (self.cell_states[r - 1][c + 1] in eligible_cell_state):
            # get neighbour in NorthEast
            eligible_neighbours.append((r - 1, c + 1))
        if (self.index_exists(r + 1, c - 1)) and (self.cell_states[r + 1][c - 1] in eligible_cell_state):
            # get neighbour in SouthWest
            eligible_neighbours.append((r + 1, c - 1))
        if (self.index_exists(r + 1, c + 1)) and (self.cell_states[r + 1][c + 1] in eligible_cell_state):
            # get neighbour in SouthEast
            eligible_neighbours.append((r + 1, c + 1))
        return eligible_neighbours
