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
    """ Reads  tai12a.dat and compares A,B matrices to previous results """
    def test_read_dat_file(self):
        A_true = np.array([[ 0, 27, 85,  2,  1, 15, 11, 35, 11, 20, 21, 61],
                            [27,  0, 80, 58, 21, 76, 72, 44, 85, 94, 90, 51],
                            [85, 80,  0,  3, 48, 29, 90, 66, 41, 15, 83, 96],
                            [ 2, 58,  3,  0, 74, 45, 65, 40, 54, 83, 14, 71],
                            [ 1, 21, 48, 74,  0, 77, 36, 53, 37, 26, 87, 76],
                            [15, 76, 29, 45, 77,  0, 91, 13, 29, 11, 77, 32],
                            [11, 72, 90, 65, 36, 91,  0, 87, 67, 94, 79,  2],
                            [35, 44, 66, 40, 53, 13, 87,  0, 10, 99, 56, 70],
                            [11, 85, 41, 54, 37, 29, 67, 10,  0, 99, 60,  4],
                            [20, 94, 15, 83, 26, 11, 94, 99, 99,  0, 56,  2],
                            [21, 90, 83, 14, 87, 77, 79, 56, 60, 56,  0, 60],
                            [61, 51, 96, 71, 76, 32,  2, 70,  4,  2, 60,  0]])
        B_true = np.array([[ 0, 21, 95, 82, 56, 41,  6, 25, 10,  4, 63,  6],
                            [21,  0, 44, 40, 75, 79,  0, 89, 35,  9,  1, 85],
                            [95, 44,  0, 84, 12,  0, 26, 91, 11, 35, 82, 26],
                            [82, 40, 84,  0, 69, 56, 86, 45, 91, 59, 18, 76],
                            [56, 75, 12, 69,  0, 39, 18, 57, 36, 61, 36, 21],
                            [41, 79,  0, 56, 39,  0, 71, 11, 29, 82, 82,  6],
                            [ 6,  0, 26, 86, 18, 71,  0, 71,  8, 77, 74, 30],
                            [25, 89, 91, 45, 57, 11, 71,  0, 89, 76, 76, 40],
                            [10, 35, 11, 91, 36, 29,  8, 89,  0, 93, 56,  1],
                            [ 4,  9, 35, 59, 61, 82, 77, 76, 93,  0, 50,  4],
                            [63,  1, 82, 18, 36, 82, 74, 76, 56, 50,  0, 36],
                            [ 6, 85, 26, 76, 21,  6, 30, 40,  1,  4, 36,  0]])
        A,B = qap_cqm.read_problem_dat('QAPLIB_data/tai12a.dat')
        self.assertEqual(A,A_true)
        self.assertEqual(B,B_true)

    def test_read_sln_file(self):
        """ Reads  wil100.sln and compares energy to known value """
        self.assertEqual(qap_cqm.read_solution('QAPLIB_solutions/wil100.sln'),273038)

    def test_cqm_variables(self):
        """ Creates 2 CQM objects (one w/ and one w/o presolve applied) and checks the variables against previous results """
        no_presolve_variables = dimod.Variables(['x0_0', 'x0_1', 'x0_2', 'x0_3', 'x0_4', 'x0_5', 'x0_6', 'x0_7', 'x0_8', 'x0_9', 'x0_10', 'x0_11',
                                                'x1_0', 'x1_1', 'x1_2', 'x1_3', 'x1_4', 'x1_5', 'x1_6', 'x1_7', 'x1_8', 'x1_9', 'x1_10', 'x1_11',
                                                'x2_0', 'x2_1', 'x2_2', 'x2_3', 'x2_4', 'x2_5', 'x2_6', 'x2_7', 'x2_8', 'x2_9', 'x2_10', 'x2_11',
                                                'x3_0', 'x3_1', 'x3_2', 'x3_3', 'x3_4', 'x3_5', 'x3_6', 'x3_7', 'x3_8', 'x3_9', 'x3_10', 'x3_11', 
                                                'x4_0', 'x4_1', 'x4_2', 'x4_3', 'x4_4', 'x4_5', 'x4_6', 'x4_7', 'x4_8', 'x4_9', 'x4_10', 'x4_11', 
                                                'x5_0', 'x5_1', 'x5_2', 'x5_3', 'x5_4', 'x5_5', 'x5_6', 'x5_7', 'x5_8', 'x5_9', 'x5_10', 'x5_11', 
                                                'x6_0', 'x6_1', 'x6_2', 'x6_3', 'x6_4', 'x6_5', 'x6_6', 'x6_7', 'x6_8', 'x6_9', 'x6_10', 'x6_11', 
                                                'x7_0', 'x7_1', 'x7_2', 'x7_3', 'x7_4', 'x7_5', 'x7_6', 'x7_7', 'x7_8', 'x7_9', 'x7_10', 'x7_11', 
                                                'x8_0', 'x8_1', 'x8_2', 'x8_3', 'x8_4', 'x8_5', 'x8_6', 'x8_7', 'x8_8', 'x8_9', 'x8_10', 'x8_11', 
                                                'x9_0', 'x9_1', 'x9_2', 'x9_3', 'x9_4', 'x9_5', 'x9_6', 'x9_7', 'x9_8', 'x9_9', 'x9_10', 'x9_11', 
                                                'x10_0', 'x10_1', 'x10_2', 'x10_3', 'x10_4', 'x10_5', 'x10_6', 'x10_7', 'x10_8', 'x10_9', 'x10_10', 'x10_11', 
                                                'x11_0', 'x11_1', 'x11_2', 'x11_3', 'x11_4', 'x11_5', 'x11_6', 'x11_7', 'x11_8', 'x11_9', 'x11_10', 'x11_11'])
        presolve_variables = dimod.Variables(range(0, 144))
        A,B = qap_cqm.read_problem_dat('QAPLIB_data/tai12a.dat')
        cqm_presolve = qap_cqm.build_cqm(A,B, pre_solve = True)
        cqm_no_presolve = qap_cqm.build_cqm(A,B, pre_solve = False)
        self.assertEqual(cqm_presolve.variables, presolve_variables)
        self.assertEqual(cqm_no_presolve.variables, no_presolve_variables)