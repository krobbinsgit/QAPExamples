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
# Written by Ken Robbins & Daniel Mahler with legacy code from John McFarland

from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel, Binary, BINARY, Integer, quicksum
from dwave.system import LeapHybridCQMSampler
from dwave.preprocessing.presolve import Presolver
import numpy as np
import cytoolz as tl
import click
import itertools as it
from os.path import dirname, join



def read_problem_dat(path:str):
    """
    Reads the .dat files from QAPLIB and converts them to numpy arrays that describe the problem.
    A and B are (n x n)-dimensional matrices that represent either 'flow' or 'distance' in a QAP
    Inputs:
        path (str): the filepath to the dat file
    Outputs:
        A (numpy array): the flow or distance matrix for the QAP described in the .dat file
        A (numpy array): the distance or flow matrix for the QAP described in the .dat file
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


def read_solution(filename:str):
    with open(filename) as solution_file:
        solution_value = int(solution_file.readline().split(' ')[-1][:-1])
    return(solution_value)


def set_qap_objective(cqm, A, B):
    """
    Writes the corresponding objective function from A and B into the CQM object
    Inputs:
        cqm (dimod.ConstrainedQuadraticModel):
        A (numpy.array): the flow or distance matrix for the QAP 
        B (numpy.array): the distance or flow matrix for the QAP
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
    Writes a one-hot constraint into the CQM object
    A one-hot constraint is such that (var_1+var_2+var_3+...+var_N = 1) for binary variables var_n
    Inputs:
        cqm (dimod.ConstrainedQuadraticModel): the CQM object representing the Quadratic Assignment Problem
        vars: the variable names that are being summed over
        label: the label for the constraint
    Outputs:
        N/A 
    `"""
    return cqm.add_constraint_from_iterable(((v, 1) for v in vars), '==', 1, label=label)


def add_discrete(cqm, vars, label):
    """
    Manually writes in a one-hot constraint to the CQM object and marks it as one-hot ("discrete")
    Faster than cqm.add_discrete for long 'vars'
    Inputs:
        cqm (dimod.ConstrainedQuadraticModel): the CQM object representing the Quadratic Assignment Problem
        vars: the variable names that are being summed over
        label: the label for the constraint
    Outputs:
        N/A 
    `"""
    lbl = add_1_hot(cqm, vars, label)
    cqm.constraints[lbl].lhs.mark_discrete()


def build_cqm(A, B, swap='auto',pre_solve:bool = True):
    """
    Constructs a CQM object which represents the Quadratic Assignment Problem.
    This includes creating variables, constructing the objective function and adding in the constraints
    Inputs:
        A (numpy.array): the flow or distance matrix for the QAP
        B (numpy.array): the distance or flow matrix for the QAP
        swap (Boolean or 'auto' = 'auto'): Whether to swap the order that the A and B matrices are checked for
                            zeros.  Most instances have a sparser "B" matrix, so we check that
                            first by default.  Instance tai256c has a sparser A matrix, so
                            'auto' swaps on that instance.
        pre_solve (Boolean = True): set to False to turn D-Wave's presolve methods off
    Outputs:
        reduced_cqm (dimod.ConstrainedQuadraticModel): a simplified CQM object representing the QAP described by A, B
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
    set_qap_objective(cqm, A, B) # A, B were reversed for some reason. Shouldn't matter

    if pre_solve==True:
        presolve = Presolver(cqm)
        presolve.load_default_presolvers()
        presolve.apply()
        reduced_cqm = presolve.detach_model()
    else:
        reduced_cqm = cqm
    return reduced_cqm

# Command line functionality
DEFAULT_PATH = join(dirname(__file__), 'QAPLIB_data', 'tai12a.dat')


@click.command(help='Solve a QAP using '
                    'LeapHybridCQMSampler.')
@click.option('--filename', type=click.Path(), default = 'tai12a',
              help=f'Path to problem file.  Default is tai12a')
@click.option('--verbose/--not_verbose', default = True,
              help='Prints information during and after the solve. Use not_verbose to turn it off')
@click.option('--pre_solve', default = True,
              help='Set pre_solve to False to turn it off')

def main(filename:str, verbose = True, pre_solve = True):
    """
    Solves the Quadratic Assignment Problem with the designation filename, then prints results and compares to QAPLIB
    Inputs:
        filename (str): the name of the QAP to read and solve, e.g. tai12a
        verbose (Boolean = True): set to False to turn off printed status updates mid-solve
        pre_solve (Boolean = True): set to False to turn D-Wave's presolve methods off
    Outputs:
        (solution_value,best_value,best.sample) (tuple):
            solution_value (int): the best-known solution according to QAPLIB
            best_value (int): the best solution value from the hybrid sampler
            best.sample (dict): the best solution variables from our hybrid sampler 
                keys are variable names
                values are 0.0 or 1.0
    """
    if verbose:
        print(f'\nBeginning to read files for {filename}\n')
    A,B = read_problem_dat(f'QAPLIB_data/{filename}.dat') # distance and flow matrices
    size = len(A)
    solution_value = read_solution(f'QAPLIB_solutions/{filename}.sln')
    
    if verbose:
        print('Making the CQM object\n')
    cqm = build_cqm(A,B,pre_solve=pre_solve)
    sampler=LeapHybridCQMSampler()
    min_time = sampler.min_time_limit(cqm) # Estimates the minimum recommended time for the CQM sampler
    if min_time >= 5:
        new_time_spent = round_decimals_up(min_time,1)
        if verbose:
            print(f'Adjusting runtime to minimum required: {new_time_spent}s\n')
    else:
        new_time_spent = 5

    if verbose:
        print('Beginning to sample\n')
    sample_set = sampler.sample_cqm(cqm, time_limit = new_time_spent)

    if verbose:
        print('Finished sampling. Beginning to filter for feasibility\n')
    feasible_samples = sample_set.filter(lambda d: d.is_feasible)
    if verbose:
        print('Finished filtering\n')
    if len(feasible_samples)>0:
        best = feasible_samples.lowest().first
        best_value = best.energy
        if best_value == int(best_value):
            best_value = int(best_value)
        if verbose:
            print(f'Feasible solution found!\n\nThe energy calculated in {new_time_spent}s is {best_value}\n')
        if (best_value == solution_value) and verbose:
            print('This is the same value as the best-known solution according to QAPLIB.\n')
        elif (best_value > solution_value) and verbose:
            print(f'This is a {relative_error_percent(best_value,solution_value)}% worse solution than the best known on QAPLIB: {solution_value}\n')
        elif (best_value < solution_value) and verbose:
            print(f'This is a {relative_error_percent(best_value,solution_value)}% better solution than the best known on QAPLIB: {solution_value}\n')
        return((solution_value,best_value,best.sample))
    else:
        print(f'No feasible solution found after {new_time_spent}s')
    

def relative_error_percent(observed:(int or float), expected:(int or float)):
    """Calculates relative error of observed to expected data up to 2 decimal places"""
    return(abs(np.round(100*(observed-expected)/expected,2)))


def round_decimals_up(number:float, decimals:int = 2):
    """
    Returns a value rounded up to a specific number of decimal places.
    Derived from kodify.net to save time
    """
    if decimals == 0:
        return np.ceil(number)
    factor = 10 ** decimals
    return np.ceil(number * factor) / factor


if __name__ == "__main__":
    main()
