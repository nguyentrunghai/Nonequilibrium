
from __future__ import print_function

import numpy as np

from MMTK import Units
from MMTK.PDB import PDBOutputFile
from MMTK.Environment import NoseThermostat
from MMTK.Dynamics import VelocityVerletIntegrator
from LangevinDynamics import LangevinIntegrator
from MMTK.Minimization import ConjugateGradientMinimizer 
from MMTK.Dynamics import TranslationRemover

from _mmtk import start_universe, get_ff
from _mmtk import end_2_end_harmonic_restraint, restraint_atoms_to_positions, com_2_x_0_0_harmonic_restraint


class MD(object):
    def __init__(self, system="ALA",
                 main_force_field="Amber12SB",
                 initial_pdb="crd.pdb",
                 molecule_indices=None,
                 initial_lambda=0.,
                 lambda_k=3000.,
                 ref_pdb_for_host_restraint="ref.pdb",
                 host_restraint_k=20000.,
                 restrain_host_carbons_only=True,
                 steps_per_iteration=1000,
                 dt=0.001,
                 temperature=300.0,
                 thermostat=None,
                 out_prefix="conf_"):
        """
        thermostat is None, "Nose" or "Langevin"
        """
        if molecule_indices is None:
            molecule_indices = {"guest": 0, "host": 1}

        self._system = system
        self._universe = start_universe(initial_pdb)
        self._main_ff = get_ff(main_force_field)

        self._host_restraint_ff  = self._get_restraint_atoms_to_positions_ff(molecule_indices["host"],
                                            ref_pdb_for_host_restraint, host_restraint_k, restrain_host_carbons_only)
        
        if self._system == "CUC":
            self._lambda_restraint_ff = self._get_restraint_com_to_x00_ff(molecule_indices["guest"],
                                                                          initial_lambda, lambda_k)

        elif self._system == "ALA":
            self._lambda_restraint_ff = self._get_end_2_end_harmonic_restraint(initial_lambda, lambda_k)

        else:
            raise RuntimeError("unknown system "+ self._system)

        self._temperature  = temperature*Units.K
        self._set_temperature()
        self._set_ffs()

        self._steps_per_iteration = steps_per_iteration
        self._dt = dt*Units.ps
        self._out_prefix = out_prefix

        self._integrator = self._get_integrator(thermostat)

    def _set_temperature(self):
        self._universe.initializeVelocitiesToTemperature(self._temperature)
        return None

    def _get_restraint_atoms_to_positions_ff(self, restrained_molecule_index, ref_pdb_for_restraint, k,
                                             restrain_carbons_only):
        if self._system == "ALA":
            return None

        elif self._system == "CUC":
            restr = restraint_atoms_to_positions(self._universe, restrained_molecule_index, ref_pdb_for_restraint, k,
                                                 restrain_carbons_only)
            return restr.get_ff()

        else:
            raise RuntimeError(self._system + " is unknown system")

    def _get_restraint_com_to_x00_ff(self, restrained_molecule_index, x_ref, k):
        assert self._system == "CUC", "system is not CUC"
        restr = com_2_x_0_0_harmonic_restraint(self._universe, restrained_molecule_index, k)
        return restr.get_ff(x_ref)

    def _get_end_2_end_harmonic_restraint(self, ref_lambda, k):
        assert self._system == "ALA", "system is not ALA"
        restr = end_2_end_harmonic_restraint(self._universe, k)
        return restr.get_ff(ref_lambda)

    def _set_ffs(self):
        ff = self._main_ff
        if self._host_restraint_ff is not None:
            ff = ff + self._host_restraint_ff

        ff = ff + self._lambda_restraint_ff
            
        self._universe.setForceField(ff)
        return None

    def _get_Langevin_integrator(self, friction_fac=0.5):
        friction = self._universe.masses() * friction_fac / Units.ps
        integrator = LangevinIntegrator(self._universe, delta_t=self._dt, friction=friction,
                                        temperature=self._temperature)
        return integrator

    def _get_integrator(self, thermostat):
        assert thermostat in [None, "Nose", "Langevin"], "Wrong thermostat: " + str(thermostat)

        if thermostat is None:
            return VelocityVerletIntegrator(self._universe, delta_t=self._dt)

        elif thermostat == "Nose":
            self._universe.thermostat = NoseThermostat(self._temperature)
            return VelocityVerletIntegrator(self._universe, delta_t=self._dt)

        elif thermostat == "Langevin":
            return self._get_Langevin_integrator()

    def _get_kinetic_energy(self):
        return self._universe.kineticEnergy()

    def _get_potential_energy(self):
        return self._universe.energy()

    def _get_coordinates(self):
        return np.array( self._universe.configuration().array )

    def _get_natoms(self):
        return self._universe.numberOfAtoms()

    def _write_pdb(self, file):
        handle = PDBOutputFile(file)
        handle.write( self._universe, self._universe.configuration() )
        handle.close()
        return None

    def minimize(self, out_pdb="min.pdb"):
        minimizer = ConjugateGradientMinimizer(self._universe)
        minimizer(convergence=1.e-3, steps=10000)
        self._write_pdb(out_pdb)
        return None

    def equilibrate(self, steps=1000):
        print("Equilibrate for ", steps, " steps")
        actions = []
        if self._system == "ALA":
            actions = [TranslationRemover(0, None, self._steps_per_iteration)]
        self._integrator(steps=steps, actions=actions)
        return None

    def propagate(self, niterations=100):
        actions = []
        if self._system == "ALA":
            actions = [TranslationRemover(0, None, self._steps_per_iteration) ]

        for iteration in range(niterations):
            self._integrator(steps=self._steps_per_iteration, actions=actions)
            print("Iteration", iteration, "potential", self._get_potential_energy(),
                  "kinetic", self._get_kinetic_energy())

            out_pdb = self._out_prefix + "%d.pdb"%iteration
            self._write_pdb(out_pdb)
        return None
