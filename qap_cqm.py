# Copyright 2023 D-Wave Systems
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel, Binary, BINARY, Integer, quicksum
from dwave.system import LeapHybridCQMSampler
# from dimod import ConstrainedQuadraticModel, Binary, Integer, quicksum
from dwave.preprocessing.presolve import Presolver
import numpy as np
import random
from itertools import product
import itertools as it
import cytoolz as tl


import time

def relative_error_percent(observed,expected):
    """Calculates relative error of observed to expected data up to 2 decimal places"""
    return(abs(np.round(100*(observed-expected)/expected,2)))

def read_problem_dat(path):
    """
    Reads the .dat files from QAPLIB and converts them to numpy arrays that describe the problem.
    A and B are (n x n)-dimensional matrices that represent either 'flow' or 'distance' in a QAP problem
    Inputs:
        path (str): the filepath to the dat file
    Outputs:
        A (numpy array): the flow or distance matrix for the QAP problem described in the .dat file
        A (numpy array): the distance or flow matrix for the QAP problem described in the .dat file
    """
    def read_matrix(lines): 
        data = []
        for line in lines:
            data.append([int(x) for x in line.split()])
        return np.array(data)
    
    with open(path, 'r') as f:
        lines = (line.strip() for line in f)
        lines = list(line for line in lines if line)

    n = int(lines[0])
    A = read_matrix(lines[1:1+n])
    B = read_matrix(lines[1+n:])
    assert len(A) == len(B) == n
    assert np.size(A,1) == np.size(B,1) == n

    return A, B

def read_solution(file_name):
    with open(file_name) as solution_file:
        solution_value = int(solution_file.readline().split(' ')[-1][:-1])
    return(solution_value)

def set_qap_objective(cqm, A, B):
    """
    Writes the corresponding objective function from A and B into the cqm object
    Inputs:
        cqm (dimod.ConstrainedQuadraticModel):
        A (numpy.array): the flow or distance matrix for the QAP problem 
        B (numpy.array): the distance or flow matrix for the QAP problem
    Outputs:
        N/A 
    """
    n = len(A)
    x = [[f'x{i}_{j}' for j in range(n)] for i in range(n)]

    cqm.add_variables(BINARY, tl.concat(x))

    for i, row in  enumerate(x):
        add_discrete(cqm, row, f'discrete_row_{i}')

    for j in range(n):
        add_1_hot(cqm, (row[j] for row in x), f'one_hot_col_{j}')

    cqm.set_objective(
        (x0, x1, a * b)
        for (i0, x_i0), (i1, x_i1) in it.product(enumerate(x), repeat=2)
        if (a := A[i0, i1])
        for (j0, x0), (j1, x1) in it.product(enumerate(x_i0), enumerate(x_i1))
        if (b := B[j0, j1])
    )

def add_1_hot(cqm, vars, label):
    """
    Writes a one-hot constraint into the cqm object
    Inputs:
        cqm (dimod.ConstrainedQuadraticModel):
        vars
        label
    Outputs:
        N/A 
    `"""
    return cqm.add_constraint_from_iterable(((v, 1) for v in vars), '==', 1, label=label)

def add_discrete(cqm, vars, label):
    """
    Manually writes in a constraint and marks it as one-hot ('discrete')
    Faster than cqm.add_discrete for long 'vars'
    Inputs:
        cqm (dimod.ConstrainedQuadraticModel):
        vars
        label
    Outputs:
        N/A 
    `"""
    lbl = add_1_hot(cqm, vars, label)
    cqm.constraints[lbl].lhs.mark_discrete()

def build_cqm(A, B, swap='auto',pre_solve=True): # Ripped from John McFarland's stuff. Takes in flow and distance matrices and outputs a cqm object
    """

    Inputs:
        A (numpy.array): the flow or distance matrix for the QAP problem 
        B (numpy.array): the distance or flow matrix for the QAP problem
        swap (Boolean or 'auto' = 'auto'): Whether to swap the order that the A and B matrices are checked for
                            zeros.  Most instances have a sparser "B" matrix, so we check that
                            first by default.  But the 256 instance has a sparser A matrix, so
                            'auto' swaps on that instance.
        pre_solve (Boolean = True): set to False to turn D-Wave's presolve methods off
    Outputs:
        reduced_cqm (dimod.ConstrainedQuadraticModel): a simplified CQM object representing the QAP problem described by A, B
    """
    n = len(A)
    if swap == 'auto':
        swap = (n == 256)
    if swap:
        tmp = A
        A = B
        B = tmp
        del tmp

    cqm = ConstrainedQuadraticModel() # Builds an empty CQM object
    set_qap_objective(cqm, B, A)

    if pre_solve==True:
        presolve = Presolver(cqm)
        presolve.load_default_presolvers()
        presolve.apply()
        reduced_cqm = presolve.detach_model()
    else:
        reduced_cqm = cqm
    return reduced_cqm

def round_decimals_up(number:float, decimals:int=2): # ripped from kodify.net
    """
    Returns a value rounded up to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return np.ceil(number)

    factor = 10 ** decimals
    return np.ceil(number * factor) / factor

def qap_solver(file_name,pre_solve=True):
    """
    Solves the 
    Inputs:
        file_name (str): the name of the QAP problem to read and solve, e.g. tai12a
        pre_solve (Boolean = True): set to False to turn D-Wave's presolve methods off
    Outputs:
        (solution_value,best_value,best.sample) (tuple):
            solution_value (int): the best-known solution according to QAPLIB
            best_value (int): the best solution value from the hybrid sampler
            best.sample (dict): the best solution variables from our hybrid sampler 
                keys are variable names
                values are 0.0 or 1.0
    """
    print('Beginning to read files.\n')
    A,B = read_problem_dat(f'QAPLIB_data/{file_name}.dat') # distance and flow matrices
    size = len(A)
    solution_value = read_solution(f'QAPLIB_solutions/{file_name}.sln')
    
    print('Making the CQM object.\n')
    cqm = build_cqm(A,B,pre_solve=pre_solve)
    sampler=LeapHybridCQMSampler()
    min_time = sampler.min_time_limit(cqm)
    if min_time >= 5:
        new_time_spent = round_decimals_up(min_time,1)
        print(f'Adjusting runtime to minimum required: {new_time_spent}s\n')
    else:
        new_time_spent = 5

    print('Beginning to sample\n')
    sample_set = sampler.sample_cqm(cqm, time_limit = new_time_spent)

    print('Finished sampling. Beginning to filter for feasibility\n')
    feasible_samples = sample_set.filter(lambda d: d.is_feasible)
    print('Finished filtering\n')
    if len(feasible_samples)>0:
        best = feasible_samples.lowest().first
        best_value = best.energy
        print(f'Feasible solution found!\n\nThe energy calculated in {new_time_spent}s is {best_value}\n')
        if best_value == solution_value:
            print('This is the same value as the best-known solution according to QAPLIB.\n')
        elif best_value >= solution_value:
            print(f'This is a {relative_error_percent(best_value,solution_value)}% worse solution than the best known on QAPLIB: {solution_value}\n')
        else:
            print(f'This is a {relative_error_percent(best_value,solution_value)}% better solution than the best known on QAPLIB: {solution_value}\n')
        return((solution_value,best_value,best.sample))
    else:
        print(f'No feasible solution found after {new_time_spent}s')
    



qap_solver('tai20a')
    
