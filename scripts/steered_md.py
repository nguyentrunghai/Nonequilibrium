
from __future__ import print_function

import numpy as np
import netCDF4 as nc

from MMTK import Units
from MMTK.PDB import PDBOutputFile
from MMTK.Environment import NoseThermostat
from MMTK.Dynamics import VelocityVerletIntegrator
from MMTK.Dynamics import TranslationRemover

from LangevinDynamics import LangevinIntegrator

from _mmtk import start_universe, get_ff
from _mmtk import end_2_end_harmonic_restraint, com_2_x_0_0_harmonic_restraint, restraint_atoms_to_positions


class SMD(object):
    def __init__(self, system="ALA",
                 main_force_field="Amber12SB",
                 initial_pdb="crd.pdb",
                 molecule_indices=None,
                 lambda_k=3000.,
                 ref_pdb_for_host_restraint="ref.pdb",
                 host_restraint_k=20000.,
                 restrain_carbons_only=True,
                 dt=0.001,
                 temperature=300.0,
                 thermostat=None,
                 out_nc_file="out.nc"):
        """
        system is either "ALA" or "CUC"
        thermostat is None, "Nose" or "Langevin"
        """
        if molecule_indices is None:
            molecule_indices = {"guest":0, "host":1}

        # system name
        self._system = system

        # create MMTK universe
        self._universe = start_universe(initial_pdb)

        # AMBER FF for the molecules
        self._main_ff = get_ff(main_force_field)

        # restraint ff for the host
        self._host_restraint_ff = self._get_restraint_atoms_to_positions_ff(molecule_indices["host"],
                                                ref_pdb_for_host_restraint, host_restraint_k, restrain_carbons_only)

        # FF that does not change with time (not the pulling harmonic),
        # equal the main ff; plus _host_restraint_ff if system is CUC
        self._fixed_ff = self._get_fixed_ff()

        # harmonic force constant for lambda (programed protocol), MMTK unit (kJ/mol / nm^2)
        self._lambda_k  = lambda_k

        self._dt = dt * Units.ps
        self._temperature = temperature * Units.K
        self._set_temperature()

        self._lambda_restraint = self._get_lambda_restraint(molecule_indices["guest"])
        self._restrained_atoms = self._lambda_restraint.get_restrained_atoms()

        self._integrator = self._get_integrator(thermostat)
        self._translational_removal = self._get_translational_removal()

        self._nc_handle = self._create_nc_handle(out_nc_file)

    def _get_restraint_atoms_to_positions_ff(self, restrained_molecule_index,
                                             ref_pdb_for_restraint, k, restrain_carbons_only):
        if self._system == "ALA":
            return None

        elif self._system == "CUC":
            restr = restraint_atoms_to_positions(self._universe, restrained_molecule_index,
                                                 ref_pdb_for_restraint, k, restrain_carbons_only)
            return restr.get_ff()

        else:
            raise RuntimeError(self._system + " is unknown system")

    def _get_fixed_ff(self):
        ff = self._main_ff
        if self._host_restraint_ff is not None:
            ff = ff + self._host_restraint_ff
        return ff

    def _set_temperature(self):
        self._universe.initializeVelocitiesToTemperature(self._temperature)
        return None

    def _get_lambda_restraint(self, restrained_molecule_index):
        if self._system == "ALA":
            return end_2_end_harmonic_restraint(self._universe, self._lambda_k)
        elif self._system == "CUC":
            return com_2_x_0_0_harmonic_restraint(self._universe, restrained_molecule_index, self._lambda_k) 
        else:
            raise RuntimeError("unknown "+ self._system)

    def _set_ffs(self):
        lambda_restraint_ff = self._lambda_restraint.get_ff(self._current_l)
        self._universe.setForceField(self._fixed_ff + lambda_restraint_ff)
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

    def _get_translational_removal(self):
        actions = []
        if self._system == "ALA":
            actions = [TranslationRemover(0, None, 1) ]
        return actions

    def _get_kinetic_energy(self):
        return self._universe.kineticEnergy()

    def _get_potential_energy(self):
        return self._universe.energy()

    def _get_positions(self):
        return np.array( self._universe.configuration().array )

    def _get_natoms(self):
        return self._universe.numberOfAtoms()

    def _get_restrained_coordinate(self):
        if self._system == "ALA":
            atom1, atom2 = self._restrained_atoms 
            return self._universe.distance(atom1, atom2)
        elif self._system == "CUC":
            com = self._restrained_atoms.centerOfMass()
            return com.array[0]
        else:
            raise RuntimeError("Unknown system "+self._system)

    def _create_nc_handle(self, out_nc_file):
        nc_handle = nc.Dataset(out_nc_file, mode="w", format="NETCDF4")

        nc_handle.createDimension("one", 1)
        nc_handle.createDimension("three", 3)
        natoms = self._get_natoms()
        nc_handle.createDimension("natoms", natoms)

        nc_handle.createDimension("unlimited_pos", None)
        nc_handle.createDimension("unlimited", None)

        nc_handle.createVariable("positions", "f8", ("unlimited_pos", "natoms", "three"))
        nc_handle.createVariable("position_time_index", "i8", ("unlimited_pos"))

        nc_handle.createVariable("tot_energies", "f8", ("unlimited"))
        nc_handle.createVariable("pot_energies", "f8", ("unlimited"))
        nc_handle.createVariable("lambda", "f8", ("unlimited"))
        nc_handle.createVariable("restrained_coordinate", "f8", ("unlimited"))

        nc_handle.createVariable("lambda_k", "f8", ("one"))
        nc_handle.variables["lambda_k"][0] = self._lambda_k

        return nc_handle

    def _save_positions(self, save_index, time_index):
        self._nc_handle.variables["positions"][save_index, :, :] = self._get_positions()
        self._nc_handle.variables["position_time_index"][save_index] = time_index
        return None

    def propagate(self, programmed_lambdas, position_save_freq=1000):
        """
        :param programmed_lambdas: list of 1d ndarray
        :param position_save_freq: int
        :return: None
        """
        programmed_lambdas = np.asanyarray(programmed_lambdas)
        steps = programmed_lambdas.shape[0]
        print("propagate for", steps, "steps")

        pot_energies = np.zeros([steps], dtype=float)
        tot_energies = np.zeros([steps], dtype=float)
        lambda_t = np.zeros([steps], dtype=float)
        restrained_coordinates = np.zeros([steps], dtype=float)

        # first step
        self._current_l = programmed_lambdas[0]
        self._set_ffs()

        pot_e = self._get_potential_energy()
        kinetic_e = self._get_kinetic_energy()
        pot_energies[0] = pot_e
        tot_energies[0] = pot_e + kinetic_e
        lambda_t[0] = self._current_l
        restrained_coordinates[0] = self._get_restrained_coordinate()

        positions_save_index = 0
        self._save_positions(positions_save_index, 0)

        print("step 0;  potential", pot_e, ";  kinetic", kinetic_e, ";  lambda", self._current_l)

        # the next (steps-1) steps
        for step in xrange(1, steps):
            self._current_l = programmed_lambdas[step]
            self._set_ffs()
            self._integrator(steps=1, actions=self._translational_removal)

            pot_e = self._get_potential_energy()
            kinetic_e = self._get_kinetic_energy()

            pot_energies[step] = pot_e
            tot_energies[step] = pot_e + kinetic_e
            lambda_t[step] = self._current_l
            restrained_coordinates[step] = self._get_restrained_coordinate()

            if ((step+1) % position_save_freq) == 0:
                positions_save_index += 1
                self._save_positions(positions_save_index, step)

                print("step", step, ";  potential", pot_e, ";  kinetic", kinetic_e, ";  lambda", self._current_l)
        print("step", step, ";  potential", pot_e, ";  kinetic", kinetic_e, ";  lambda", self._current_l)

        self._nc_handle.variables["tot_energies"][:] = tot_energies
        self._nc_handle.variables["pot_energies"][:] = pot_energies
        self._nc_handle.variables["lambda"][:] = lambda_t
        self._nc_handle.variables["restrained_coordinate"][:] = restrained_coordinates

        self._nc_handle.close()

        return None

    def write_pdb(self, file_name):
        """
        :param file_name: str
        :return:
        """
        handle = PDBOutputFile(file_name)
        handle.write(self._universe, self._universe.configuration())
        handle.close()
        return None


