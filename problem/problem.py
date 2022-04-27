"""
    Sample Run:
    python .py
"""
import os, sys
sys.path.append(os.getcwd())

import numpy as np

from pymoo.core.problem import Problem

class FON(Problem):

    def __init__(self, n_var= 2):
        self.n_var = n_var
        super().__init__(n_var= self.n_var,
                         n_obj= 2,
                         n_constr= 0,
                         xl= -100 * np.ones((self.n_var, )),
                         xu=  100 * np.ones((self.n_var, ))
                         )

    def _evaluate(self, x, out, *args, **kwargs):
        # Objective Functions
        obj1     = 1 - np.exp( - np.sum( (x - np.sqrt(1/self.n_var) )**2 ), axis= 1)
        obj2     = 1 - np.exp( - np.sum( (x + np.sqrt(1/self.n_var) )**2 ), axis= 1)
        out["F"] = np.column_stack([obj1, obj2])

        # Constraints
        out["G"] = None