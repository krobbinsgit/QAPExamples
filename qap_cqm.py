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
import matplotlib.pyplot as plt



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
        lines = solution_file.readlines()
        num_lines = len(lines)
        best_value_QAPLIB = int(lines[0].split(' ')[-1][:-1])
        solution_permutation = []
        for j in range(1,num_lines):
            solution_permutation += [int(i)-1 for i in lines[j].split()]
    return((solution_permutation,best_value_QAPLIB))


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

def solution_plotter(filename:str,solution:dict,A,B):
    """
    Plots the solution obtained by the hybrid solver
    Inputs:
        filename (str): the name of the problem that was solved, e.g. tai5a
        solution (dict): a dictionary showing the binary variable values in the hybrid solver's solution
        A (numpy.array): the flow or distance matrix for the QAP
        B (numpy.array): the distance or flow matrix for the QAP
    Outputs:
        None
    """
    n = len(A)
    vals = list(solution.values())
    x_positions = [int(j/n) for j in range(len(vals)) if vals[j] == 1.0]
    y_positions = [j%n for j in range(len(vals)) if vals[j] == 1.0]
    plt.grid(True)
    plt.xlabel('Locations')
    plt.ylabel('Stations')
    plt.title(f'Station Placement for {filename}')
    if n <= 15:
        grid_spacing = 1
    else:
        grid_spacing = np.ceil(n/10)
    plt.xticks(np.arange(0, n+1, grid_spacing))
    plt.yticks(np.arange(0, n+1, grid_spacing))
    plt.xlim(-1,n+1)
    plt.ylim(-1,n+1)
    plt.scatter(x_positions,y_positions, label = 'Hybrid sampler solution')
    plt.savefig(f'{filename}_solution_plot.png')

    
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
              help = 'Set the sampler\'s maximum runtime manually')
@click.option('--plot', default = True,
              help = 'Set to False to stop plotting the solution as a graph')

def main(filename:str, verbose = True, pre_solve = True, runtime = 5, plot = True):
    """
    Solves the Quadratic Assignment Problem with the designated filename, then prints/plots results and compares to QAPLIB
    Inputs:
        filename (str): the name of the QAP to read and solve, e.g. tai12a or wil100
        verbose (Boolean = True): set to False to turn off printed status updates mid-solve
        pre_solve (Boolean = True): set to False to turn D-Wave's presolve methods off
        runtime (float = 5): the runtime for the CQM sampler
            If runtime is too low then the code automatically adjusts runtime to the minimum required
        plot (Boolean = True): set to False to prevent plotting the solution with pyplot
    Outputs:
        (best_value_QAPLIB,  best_solution_QAPLIB_perm, best_value_hybrid, best_solution_hybrid) (tuple):
            best_value_QAPLIB (int): the best-known solution according to QAPLIB
            best_solution_QAPLIB_perm (list of ints): the problem solution read from the .sln file
            best_value_hybrid (int): the best solution value from the hybrid sampler
            best_solution_hybrid (dict): the best solution variables from our hybrid sampler 
                keys are variable names
                    If pre_solve == True then variable names will be integers from 0 to n**2
                    If pre_solve == False then variable names will be like 'xi_j' where 0 <= i,j < n
                values are only 0.0 or 1.0
    """
    # Read the problem and solution
    if verbose:
        print(f'\nBeginning to read files for {filename}')
    A,B = read_problem_dat(f'QAPLIB_problems/{filename}.dat') # distance and flow matrices
    n = len(A)
    best_solution_QAPLIB_perm, best_value_QAPLIB = read_solution(f'QAPLIB_solutions/{filename}.sln') # The best-known solutions
    if verbose:
        print(f'This problem will need to place {n} stations into {n} locations\n')
        print('Now making the CQM object\n')

    # Build CQM object
    cqm = build_cqm(A,B,pre_solve=pre_solve)

    # Specify sampler
    sampler = LeapHybridCQMSampler()

    # Pick runtime for sampler
    min_time = sampler.min_time_limit(cqm) # Estimates the minimum recommended time for the CQM sampler
    if not runtime:
        if min_time >= 5:
            new_time_spent = round_decimals_up(min_time,1)
            if verbose:
                print(f'Adjusting sampler\'s maximum runtime to minimum required: {new_time_spent}s\n')
        else:
            new_time_spent = 5
    elif runtime >= min_time: # If manual runtime is more than minimum runtime
        new_time_spent = runtime
        if verbose:
            print(f'Sampler\'s maximum runtime manually set to {new_time_spent}s\n')
    else: # If manual runtime is less than minimum runtime
        new_time_spent = round_decimals_up(min_time,1)
        if verbose:
            print(f'Manual runtime too low\nSetting new maximum runtime to minimum required: {new_time_spent}s\n')

    # Run the sampler
    if verbose:
        print('Beginning to sample')
    sample_set = sampler.sample_cqm(cqm, time_limit = new_time_spent)

    # Filter the solutions
    if verbose:
        print('Finished sampling. Beginning to filter for feasibility')
    feasible_samples = sample_set.filter(lambda d: d.is_feasible)
    
    # Analyze the solution and print results
    if len(feasible_samples)>0:
        best = feasible_samples.lowest().first
        best_value_hybrid = best.energy
        best_solution_hybrid = best.sample
        if best_value_hybrid == int(best_value_hybrid):
            best_value_hybrid = int(best_value_hybrid)
        print(f'\nFeasible solution found!\nThe total cost of the solution is {best_value_hybrid}')
        if (best_value_hybrid == best_value_QAPLIB):
            print('This is the same value as the best-known solution according to QAPLIB.\n')
        elif (best_value_hybrid > best_value_QAPLIB):
            print(f'This is a {relative_error_percent(best_value_hybrid,best_value_QAPLIB)}% higher solution from the best-known solution on QAPLIB: {best_value_QAPLIB}\n')
        elif (best_value_hybrid < best_value_QAPLIB):
            print(f'This is a {relative_error_percent(best_value_hybrid,best_value_QAPLIB)}% better value than the best-known solution on QAPLIB: {best_value_QAPLIB}\n')
        
        # Plot the solution if plot == True
        if plot == True:
            solution_plotter(filename,best_solution_hybrid,A,B)

        return((best_value_QAPLIB,  best_solution_QAPLIB_perm, best_value_hybrid, best_solution_hybrid))

    else:
        print(f'No feasible solution found after {new_time_spent}s')
        print(f'Try manually increasing the runtime for the LeapHybridCQMSampler with the \"--runtime\" option\n')

if __name__ == "__main__":
    main()
