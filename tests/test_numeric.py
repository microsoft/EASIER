# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

import torch

import easier as esr
from easier.numeric import solver
from easier.examples.models import Poisson, Circuit, Poisson1D


@pytest.mark.usefixtures('dummy_dist_env')
class TestLinearSolver:

    backend_list = ['torch', 'cpu', 'gpu'] if torch.cuda.is_available() \
        else ['torch', 'cpu']

    def test_cg(self):
        for backend in self.backend_list:
            for eqn_type in ['poi2d', 'poi1d', 'circuit']:
                # for prec_type in ['symmetric', None]:
                for prec_type in [None]:
                    if eqn_type == 'circuit' and prec_type == 'symmetric':
                        continue

                    if eqn_type == 'poi2d':
                        eqn = Poisson(100)
                        initializers = []
                    elif eqn_type == 'poi1d':
                        eqn = Poisson1D(100)
                        initializers = []
                    elif eqn_type == 'circuit':
                        eqn = Circuit()
                        initializers = []

                    sol = solver.CG(eqn.A, eqn.b, eqn.x)
                    esr.compile([sol] + initializers, backend=backend)

                    for init in initializers:
                        init()

                    info = sol.solve(atol=1e-4, maxiter=1000)
                    assert info["residual"] < 1e-4

    def test_gmres(self):
        for backend in self.backend_list:
            for eqn_type in ['poi2d', 'poi1d', 'circuit']:
                for prec_type in [None]:
                    # for prec_type in ['forward', 'backward', 'symmetric', None]:
                    if eqn_type == 'circuit' and prec_type is not None:
                        continue

                    if eqn_type == 'poi2d':
                        eqn = Poisson(100)
                        initializers = []
                    elif eqn_type == 'poi1d':
                        eqn = Poisson1D(100)
                        initializers = []
                    elif eqn_type == 'circuit':
                        eqn = Circuit()
                        initializers = []

                    sol = solver.GMRES(eqn.A, eqn.b, eqn.x)
                    esr.compile([sol] + initializers, backend=backend)

                    for init in initializers:
                        init()

                    info = sol.solve(atol=1e-4, maxiter=1000)
                    assert info["residual"] < 1e-4
