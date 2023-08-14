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
        """ Reads a matrix printed in the .dat file and returns it as a numpy array """ 
        data = []
        for line in lines:
            data.append([int(x) for x in line.split()])
        return np.array(data)
    
    with open(path, 'r') as f:
        lines = (line.strip() for line in f)
        lines = list(line for line in lines if line)

    with open(path) as solution_file:
        n = int(solution_file.readline())
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
    if swap == 'auto': # The below clause forces swapping A <-> B for tai256c, the largest problem in QAPLIB
        swap = (n == 256) # tai256c is a special case with special structure in the A, B matrices
    if swap:
        tmp = A 
        A = B
        B = tmp
        del tmp

    cqm = ConstrainedQuadraticModel() # Builds an empty CQM object
    set_qap_objective(cqm, A, B) # A, B were reversed for some reason. Shouldn't matter

    x = [[f'x{i}_{j}' for j in range(n)] for i in range(n)]

    for i, row in enumerate(x):
        add_discrete(cqm, row, f'discrete_row_{i}')
    for j in range(n):
        add_1_hot(cqm, (row[j] for row in x), f'one_hot_col_{j}')
    if pre_solve==True:
        presolve = Presolver(cqm)
        presolve.load_default_presolvers()
        presolve.apply()
        reduced_cqm = presolve.detach_model()
    else:
        reduced_cqm = cqm
    return reduced_cqm

    
def relative_error_percent(observed:(int or float), expected:(int or float)):
    """
    Calculates the relative error of observed vs. expected data to 2 decimal places
    Returns the absolute value of the relative error
    """
    return(abs(np.round(100*(observed-expected)/expected,2)))


def round_decimals_up(number:float, decimals:int = 2):
    """
    Returns a value rounded UP to a specific number of decimal places.
    Used to round estimated runtime ABOVE the minimum required by the CQM sampler
    Derived from kodify.net
    """
    if decimals == 0:
        return np.ceil(number)
    factor = 10 ** decimals
    return np.ceil(number * factor) / factor



# Command line functionality
DEFAULT_PATH = join(dirname(__file__), 'QAPLIB_problems', 'tai12a.dat')

@click.command(help = 'Solve a QAP using LeapHybridCQMSampler. Some examples include tho30, tai40a, els19, nug30 and more')
@click.option('--filename', type=click.Path(), default = 'tai12a',
              help = 'Path to problem file.  Default is tai12a')
@click.option('--verbose', default = True,
              help = 'Prints information during and after the solve. Set it to False to turn it off')
@click.option('--pre_solve', default = True,
              help = 'Set pre_solve to False to turn it off')
@click.option('--runtime', type = float,
              help = 'Set the runtime manually')

def main(filename:str, verbose = True, pre_solve = True, runtime = 5):
    """
    Solves the Quadratic Assignment Problem with the designated filename, then prints results and compares to QAPLIB
    Inputs:
        filename (str): the name of the QAP to read and solve, e.g. tai12a or wil100
        verbose (Boolean = True): set to False to turn off printed status updates mid-solve
        pre_solve (Boolean = True): set to False to turn D-Wave's presolve methods off
        runtime (float = 5): the runtime for the CQM sampler
            If runtime is too low then the code automatically adjusts runtime to the minimum required
    Outputs:
        (solution_value,best_value,best.sample) (tuple):
            solution_value (int): the best-known solution according to QAPLIB
            best_value (int): the best solution value from the hybrid sampler
            best.sample (dict): the best solution variables from our hybrid sampler 
                keys are variable names
                    If pre_solve == True then variable names will be integers from 0 to n**2
                    If pre_solve == False then variable names will be like 'xi_j' where 0 <= i,j < n
                values are only 0.0 or 1.0
    """
    if verbose:
        print(f'\nBeginning to read files for {filename}\n')
    A,B = read_problem_dat(f'QAPLIB_problems/{filename}.dat') # distance and flow matrices
    n = len(A)
    solution_value = read_solution(f'QAPLIB_solutions/{filename}.sln')
    
    if verbose:
        print('Making the CQM object\n')
    cqm = build_cqm(A,B,pre_solve=pre_solve)
    sampler=LeapHybridCQMSampler()
    min_time = sampler.min_time_limit(cqm) # Estimates the minimum recommended time for the CQM sampler
    if not runtime:
        if min_time >= 5:
            new_time_spent = round_decimals_up(min_time,1)
            if verbose:
                print(f'Adjusting runtime to minimum required: {new_time_spent}s\n')
        else:
            new_time_spent = 5
    elif runtime >= min_time: # If manual runtime is more than minimum runtime
        new_time_spent = runtime
        if verbose:
            print(f'Runtime is manually set to {new_time_spent}s\n')
    else: # If manual runtime is less than minimum runtime
        new_time_spent = round_decimals_up(min_time,1)
        if verbose:
            print(f'Manual runtime too low\nSetting new runtime to minimum required: {new_time_spent}s\n')
    if verbose:
        print('Beginning to sample')
    sample_set = sampler.sample_cqm(cqm, time_limit = new_time_spent)

    if verbose:
        print('Finished sampling. Beginning to filter for feasibility')
    feasible_samples = sample_set.filter(lambda d: d.is_feasible)
    if len(feasible_samples)>0:
        best = feasible_samples.lowest().first
        best_value = best.energy
        if best_value == int(best_value):
            best_value = int(best_value)
        print(f'\nFeasible solution found!\nThe energy calculated is {best_value}')
        if (best_value == solution_value):
            print('This is the same value as the best-known solution according to QAPLIB.\n')
        elif (best_value > solution_value):
            print(f'This is a {relative_error_percent(best_value,solution_value)}% worse value than the best-known solution on QAPLIB: {solution_value}\n')
        elif (best_value < solution_value):
            print(f'This is a {relative_error_percent(best_value,solution_value)}% better value than the best-known solution on QAPLIB: {solution_value}\n')
        return((solution_value,best_value,best.sample))
    else:
        print(f'No feasible solution found after {new_time_spent}s')
        print(f'Try manually increasing the runtime for the LeapHybridCQMSampler with the \"--runtime\" option\n')

if __name__ == "__main__":
    main()
