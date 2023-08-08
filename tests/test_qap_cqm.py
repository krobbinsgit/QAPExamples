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

import subprocess
import unittest
import os
import sys
import numpy as np

import dimod
from dwave.system import LeapHybridCQMSampler

import qap_cqm



project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestSmoke(unittest.TestCase):
    @unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")
    def test_smoke(self):
        """Run qap_cqm.py and check that nothing crashes"""

        demo_file = os.path.join(project_dir, 'qap_cqm.py')
        subprocess.check_output([sys.executable, demo_file])

class Testcqm_qap(unittest.TestCase):
    """
    Reads tai5a.dat and compares A,B matrices to previous results
    Then reads tai5a.sln and compares the value it read to a known solution value
    Then creates CQM objects with and without pre_solve and checks variables
    Then solves the presolve-enabled CQM object and ensures answer is within 10% of known optimal solution
    """
    def test_read_dat_file(self):
        A_true = np.array([[ 0,  6, 44, 40, 75], 
                           [ 6,  0, 79,  0, 89], 
                           [44, 79,  0, 35,  9], 
                           [40,  0, 35,  0,  1], 
                           [75, 89,  9,  1,  0]])
        B_true = np.array([[ 0, 21, 95, 82, 56], 
                           [21,  0, 41,  6, 25], 
                           [95, 41,  0, 10,  4], 
                           [82,  6, 10,  0, 63], 
                           [56, 25,  4, 63,  0]])
        A,B = qap_cqm.read_problem_dat('QAPLIB_data/tai5a.dat')
        no_presolve_variables = ['x0_0', 'x0_1', 'x0_2', 'x0_3', 'x0_4', 
                                 'x1_0', 'x1_1', 'x1_2', 'x1_3', 'x1_4', 
                                 'x2_0', 'x2_1', 'x2_2', 'x2_3', 'x2_4', 
                                 'x3_0', 'x3_1', 'x3_2', 'x3_3', 'x3_4', 
                                 'x4_0', 'x4_1', 'x4_2', 'x4_3', 'x4_4']
        presolve_variables = range(0, 25)
        cqm_presolve = qap_cqm.build_cqm(A,B, pre_solve = True)
        cqm_no_presolve = qap_cqm.build_cqm(A,B, pre_solve = False)
        sampler = LeapHybridCQMSampler()
        sample_set = sampler.sample_cqm(cqm_presolve, time_limit = 5)
        feasible_samples = sample_set.filter(lambda d: d.is_feasible)
        best_energy = feasible_samples.lowest().first.energy
        np.testing.assert_almost_equal(A,A_true) 
        np.testing.assert_almost_equal(B,B_true)
        self.assertEqual(qap_cqm.read_solution('QAPLIB_solutions/tai5a.sln'),12902)
        self.assertEqual(cqm_presolve.variables, presolve_variables)
        self.assertEqual(cqm_no_presolve.variables, no_presolve_variables)
        self.assertGreaterEqual(best_energy, 12902) # best_energy >= known optimal solution
        self.assertLessEqual(best_energy,14193) # best_energy <= 1.1*(known optimal solution)
