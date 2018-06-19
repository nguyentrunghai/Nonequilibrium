
from __future__ import print_function

import copy

import numpy as np
import netCDF4 as nc

from MMTK import Units
from MMTK.PDB import PDBOutputFile
from MMTK.Environment import NoseThermostat
from MMTK.Dynamics import VelocityVerletIntegrator
from MMTK.Minimization import ConjugateGradientMinimizer 
from MMTK.Dynamics import TranslationRemover
from MMTK.ParticleProperties import Configuration

from LangevinDynamics import LangevinIntegrator

from _mmtk import start_universe
from _mmtk import get_ff
from _mmtk import end_2_end_harmonic_restraint, com_2_x_0_0_harmonic_restraint, restraint_atoms_to_positions

TEMPERATURE = 300.
KB = 0.0083144621  # kJ/mol/K
BETA = 1./KB/TEMPERATURE


class UmbrellaSampling(object):
    """
    doing Umbrella Sampling with or without replica exchange
    """
    def __init__(self,
                 system="ALA",
                 main_force_field="Amber12SB",
                 initial_pdbs="coord.pdb",
                 molecule_indices=None,
                 list_of_lambdas=None,
                 lambda_k=150.,
                 ref_pdb_for_host_restraint="ref.pdb",
                 host_restraint_k=20000.,
                 restrain_host_carbons_only=True,
                 time_step=0.001,
                 temperature=300.0,
                 thermostat=None,
                 out_nc_file="out.nc"):
        """
        system                  : either "ALA" or "CUC"
        main_force_field        : AMBER ff for the molecules in system
        initial_pdbs            : str or list of str, names of pdb files, must be the same len as list_of_lambdas
        molecule_indices        : which is host, which is guest, only used for host-guest system
        list_of_lambdas         : list of restraint window centers
        lambda_k                : harmonic force constant for the restraints

        ref_pdb_for_host_restraint  : if system is "CUC" (host-guest), this pdb is ref for restraining the host
        host_restraint_k        : force constant for restraining the host
        restrain_host_carbons_only   : True or False
        thermostat              : None, "Nose" or "Langevin"
        """
        if type(initial_pdbs) == str:
            list_of_initial_pdbs = [initial_pdbs] * len(list_of_lambdas)
        elif type(initial_pdbs) == list:
            list_of_initial_pdbs = initial_pdbs
        else:
            list_of_initial_pdbs = []

        if list_of_lambdas is None:
            list_of_lambdas = []

        assert len(list_of_initial_pdbs) == len(list_of_lambdas), "len of list_of_initial_pdbs and list_of_lambdas are not the same"
        assert system in ["ALA", "CUC"], "Unknown system: " + system
        assert thermostat in [None, "Nose", "Langevin"], "Unknown thermostat: " + str(thermostat)

        if molecule_indices is None:
            molecule_indices = {"guest": 0, "host": 1}

        # system name
        self._system = system

        # create list of MMTK universes
        self._universes = [start_universe(pdb_file) for pdb_file in list_of_initial_pdbs]

        self._K = len(self._universes)

        # univserse for non-perturbed system, not simulated
        self._universe_0 = start_universe(list_of_initial_pdbs[0])
        self._set_unperturbed_ff(main_force_field, molecule_indices["host"], ref_pdb_for_host_restraint, 
                                                                    host_restraint_k, restrain_host_carbons_only )

        # AMBER FF for the molecules
        self._list_of_main_ff = [ get_ff(main_force_field) for _ in range(self._K) ]

        # restraint ff for the host
        self._list_of_host_restraint_ff  = self._get_restraint_atoms_to_positions_ff( molecule_indices["host"],
                                        ref_pdb_for_host_restraint, host_restraint_k, restrain_host_carbons_only )

        # common ff among all replica
        self._list_of_common_ff = self._get_common_ff()

        self._list_of_lambdas = list_of_lambdas
        self._lambda_k  = lambda_k

        self._time_step = time_step * Units.ps
        self._temperature = temperature * Units.K
        self._set_temperature()

        self._lambda_restraints = self._get_lambda_restraints(molecule_indices["guest"])
        self._list_of_restrained_atoms = [lambda_restraint.get_restrained_atoms() for lambda_restraint in self._lambda_restraints]
        self._set_ffs()

        self._integrators = self._get_integrators(thermostat)
        self._translational_removals = [self._get_translational_removal() for _ in range(self._K)]

        self._initialize_variables_4_exchange()

        self._out_nc_file = out_nc_file
        self._create_nc_handle()

    def _get_restraint_atoms_to_positions_ff(self, restrained_molecule_index, ref_pdb_for_restraint,
                                             k_const, restrain_host_carbons_only):
        """
        """
        if self._system == "ALA":
            return [None] * self._K

        elif self._system == "CUC":
            restr_ffs = []

            for universe in self._universes:
                restr = restraint_atoms_to_positions( universe, restrained_molecule_index, ref_pdb_for_restraint, 
                                                        k_const, restrain_host_carbons_only )
                restr_ffs.append( restr.get_ff() )

            return restr_ffs

        else:
            raise Exception("Unknown system " + self._system)

    def _get_common_ff(self):
        ffs = []
        for main_ff, host_restraint_ff in zip(self._list_of_main_ff, self._list_of_host_restraint_ff):
            if host_restraint_ff is not None:
                ffs.append(main_ff + host_restraint_ff)
            else:
                ffs.append(main_ff)
        return ffs

    def _set_unperturbed_ff(self, main_force_field, restrained_molecule_index, ref_pdb_for_restraint,
                            k_const, restrain_host_carbons_only):
        ff = get_ff(main_force_field)

        if self._system == "ALA":
            self._universe_0.setForceField(ff)

        elif self._system == "CUC":
            host_restr = restraint_atoms_to_positions(self._universe_0, restrained_molecule_index,
                                                       ref_pdb_for_restraint, k_const, restrain_host_carbons_only)
            rest_ff = host_restr.get_ff()
            self._universe_0.setForceField(ff + rest_ff)

        else:
            raise Exception("Unknown system " + self._system)
        return None

    def _set_temperature(self):
        for universe in self._universes:
            universe.initializeVelocitiesToTemperature(self._temperature)
        return None

    def _get_lambda_restraints(self, restrained_molecule_index):
        if self._system == "ALA":
            return [end_2_end_harmonic_restraint(universe, self._lambda_k) for universe in self._universes]

        elif self._system == "CUC":
            return [com_2_x_0_0_harmonic_restraint(universe, restrained_molecule_index, self._lambda_k)
                    for universe in self._universes]

        else:
            raise Exception("Unknown system "+ self._system)

    def _set_ffs(self):
        for universe, common_ff, lambda_restraint, lambda_value in zip(self._universes, self._list_of_common_ff, 
                self._lambda_restraints, self._list_of_lambdas):

            lambda_restraint_ff = lambda_restraint.get_ff(lambda_value)
            universe.setForceField(common_ff + lambda_restraint_ff)

        return None

    def _get_Langevin_integrator(self, universe, friction_fac=0.5):
        friction = universe.masses() * friction_fac / Units.ps 
        integrator = LangevinIntegrator(universe, delta_t=self._time_step, friction=friction, temperature=self._temperature)
        return integrator

    def _get_integrators(self, thermostat):
        if thermostat is None:
            return [VelocityVerletIntegrator(universe, delta_t=self._time_step) for universe in self._universes]

        elif thermostat == "Nose":
            for universe in self._universes:
                universe.thermostat = NoseThermostat(self._temperature)
            return [VelocityVerletIntegrator(universe, delta_t=self._time_step) for universe in self._universes]

        elif thermostat == "Langevin":
            return [self._get_Langevin_integrator(universe) for universe in self._universes]
        
        else:
            raise Exception("Unknown thermostat " + thermostat)

    def _get_translational_removal(self):
        actions = []
        if self._system == "ALA":
            actions = [TranslationRemover(0, None, 1) ]
        return actions

    def _get_natoms(self):
        return self._universe_0.numberOfAtoms()

    def _get_potential_energy_in_kT(self, universe):
        pot_e = universe.energy() / self._temperature / KB
        return pot_e

    def _get_confs_for_all_replicas(self):
        confs = [copy.deepcopy( universe.configuration().array ) for universe in self._universes]
        return confs

    def _set_configuration(self, universe, conf_array):
        universe.setConfiguration( Configuration( universe, conf_array ) )
        return None

    def _get_u_kl(self):
        """
        set self._u_kl    :   reduced potential energy of configuration drawn from state k and evaluated at state l
                            in kT
        """
        self._u_kl = np.zeros([self._K, self._K], dtype=float)
        confs = self._get_confs_for_all_replicas()

        for l, universe_l in enumerate(self._universes):

            for k, conf_k in enumerate(confs):
                # take conf from uk and put it in a copy of ul
                self._set_configuration(universe_l, conf_k)

                self._u_kl[k, l] = self._get_potential_energy_in_kT(universe_l)

        # set the configuration back
        for universe, conf in zip(self._universes, confs):
            self._set_configuration(universe, conf)
        return None

    def _get_u0_k(self):
        """
        set self._u0_k :   reduced potential energy of configuration drawn from state k and evaluated at unperturbed state
                        in kT
        """
        self._u0_k = np.zeros([self._K], dtype=float)
        confs = self._get_confs_for_all_replicas()

        for k, conf_k in enumerate(confs):

            self._set_configuration(self._universe_0, conf_k)
            self._u0_k[k] = self._get_potential_energy_in_kT(self._universe_0)

        return None

    def _get_positions_k(self):
        """
        set  :   self._positions_k,    shape (K, natoms, 3)    # 
                    in nm
        """
        natoms = self._get_natoms()
        self._positions_k = np.zeros([self._K, natoms, 3], dtype=float)

        for k, uk in enumerate(self._universes):
            conf_k = copy.deepcopy( uk.configuration().array )
            self._positions_k[k, :, :] = conf_k

        return None

    def _get_restrained_coordinates(self):
        """
        set   self._restrained_coordinates shape (K,)
        """
        restrained_coordinates = []

        if self._system == "ALA":
            for universe, restrained_atoms in zip(self._universes, self._list_of_restrained_atoms):
                atom1, atom2 = restrained_atoms
                restrained_coordinates.append( universe.distance(atom1, atom2) )

        elif self._system == "CUC":
            for restrained_atoms in self._list_of_restrained_atoms:
                restrained_coordinates.append( restrained_atoms.centerOfMass().array[0] )

        else:
            raise Exception("Unknown system " + self._system)

        self._restrained_coordinates = np.array(restrained_coordinates, dtype=float)
        return None

    def _initialize_variables_4_exchange(self):
        self._exchange_pairs = []
        self._exchange_pairs.append( zip( range(0, self._K, 2), range(1, self._K, 2) ) )
        self._exchange_pairs.append( zip( range(1, self._K, 2), range(2, self._K, 2) ) )
        
        self._odd_or_even = 0

        self._exchange_attempt = np.zeros([self._K - 1], dtype=int)
        self._exchange_accept  = np.zeros([self._K - 1], dtype=int)
        return None

    def _exchange_prob(self, k1, k2):
        """
        return US exchange probability between state k1 and k2
        TODO
        """
        current_e = self._u_kl[k1, k1] + self._u_kl[k2, k2]
        after_exchange_e = self._u_kl[k1, k2] + self._u_kl[k2, k1]
        de = after_exchange_e - current_e 
        return np.exp(-de)

    def _do_exchange(self, k1, k2):
        conf_k1 = copy.deepcopy( self._universes[k1].configuration().array )
        conf_k2 = copy.deepcopy( self._universes[k2].configuration().array )

        self._set_configuration(self._universes[k1], conf_k2)
        self._set_configuration(self._universes[k2], conf_k1)

        return None

    def _try_exchange_for_pairs(self):
        self._exchange_attempt = np.zeros([self._K - 1], dtype=int)
        self._exchange_accept  = np.zeros([self._K - 1], dtype=int)

        for k1, k2 in self._exchange_pairs[self._odd_or_even]:
            self._exchange_attempt[k1] = 1
            ex_prob = self._exchange_prob(k1, k2)

            if ex_prob >= 1:
                self._do_exchange(k1, k2)
                self._exchange_accept[k1] = 1

            elif ex_prob > np.random.random():
                self._do_exchange(k1, k2)
                self._exchange_accept[k1] = 1

        if self._odd_or_even == 0:
            self._odd_or_even = 1
        elif self._odd_or_even == 1:
            self._odd_or_even = 0
        return None

    def _create_nc_handle(self):
        self._nr_times_data_is_saved = 0
        self._nc_handle = nc.Dataset(self._out_nc_file, mode="w", format="NETCDF4")

        natoms = self._get_natoms()

        self._nc_handle.createDimension("one", 1)
        self._nc_handle.createDimension("three", 3)
        self._nc_handle.createDimension("nstates", self._K)
        self._nc_handle.createDimension("nstates_mnus_1", self._K - 1)
        self._nc_handle.createDimension("natoms", natoms)
        self._nc_handle.createDimension("unlimited_sim_len", None)

        self._nc_handle.createVariable( "lambda_k", "f8", ("one") )
        self._nc_handle.createVariable( "temperature", "f8", ("one") )
        self._nc_handle.createVariable( "time_step", "f8", ("one") )

        self._nc_handle.createVariable( "list_of_lambdas", "f8", ("nstates") )
        self._nc_handle.createVariable( "exchange_attempt", "i8", ("unlimited_sim_len", "nstates_mnus_1") )
        self._nc_handle.createVariable( "exchange_accept",  "i8", ("unlimited_sim_len", "nstates_mnus_1") )

        self._nc_handle.createVariable( "u_kln", "f8", ("nstates", "nstates", "unlimited_sim_len") )
        self._nc_handle.createVariable( "u0_kn", "f8", ("nstates", "unlimited_sim_len") )
        self._nc_handle.createVariable( "restrained_coordinates_kn", "f8", ("nstates", "unlimited_sim_len") )
        self._nc_handle.createVariable( "positions_nk", "f8", ("unlimited_sim_len", "nstates", "natoms", "three") )

        self._nc_handle.variables["lambda_k"][0]    = self._lambda_k
        self._nc_handle.variables["temperature"][0] = self._temperature
        self._nc_handle.variables["time_step"][0]   = self._time_step
        self._nc_handle.variables["list_of_lambdas"][:] = np.array(self._list_of_lambdas)

        return None

    def _save_data_2_nc(self):
        self._nc_handle.variables["exchange_attempt"][self._nr_times_data_is_saved, :] = self._exchange_attempt
        self._nc_handle.variables["exchange_accept"][self._nr_times_data_is_saved, :]  = self._exchange_accept

        self._nc_handle.variables["u_kln"][:, :, self._nr_times_data_is_saved] = self._u_kl
        self._nc_handle.variables["u0_kn"][:, self._nr_times_data_is_saved] = self._u0_k
        self._nc_handle.variables["restrained_coordinates_kn"][:, self._nr_times_data_is_saved] = self._restrained_coordinates
        self._nc_handle.variables["positions_nk"][self._nr_times_data_is_saved, :, :, :] = self._positions_k

        self._nr_times_data_is_saved += 1
        return None

    def minimize_all_replicas(self, convergence=1.e-3, steps=2000):
        for k, universe in enumerate(self._universes):
            print("Minimizing relica", k)
            minimizer = ConjugateGradientMinimizer(universe)
            minimizer(convergence=convergence, steps=steps)
        return None

    def propagate_all_replicas(self, steps, want_exchange=True, save_2_nc=False):
        print("Propagating all replicas for %d steps" %steps)

        for integrator, translational_removal in zip(self._integrators, self._translational_removals):
            integrator(steps=steps, actions=translational_removal)

        self._get_u_kl()                        # set self._u_kl
        self._get_u0_k()                        # set self._u0_k
        self._get_positions_k()                 # set self._positions_k
        self._get_restrained_coordinates()      # self._restrained_coordinates 

        print("Potential Energies (kT):")
        print(" ".join( [ "%15.5f" % self._u_kl[k, k] for k in range(self._K) ] ))

        if want_exchange:
            self._try_exchange_for_pairs()
            print("Accepted exchange")
            print(" ".join(["%5d" % acc for acc in self._exchange_accept]))

        if save_2_nc:
            print("Saving to " + self._out_nc_file)
            self._save_data_2_nc()

        print("\n-------------------------\n")
        return None

    def write_to_pdb(self, prefix):
        for k, universe in enumerate(self._universes):
            out_file = prefix + "_k%d.pdb" %k
            print("Writing coordinates to " + out_file)
            handle = PDBOutputFile(out_file)
            handle.write( universe, universe.configuration() )
            handle.close()
        return None

    def close_nc(self):
        self._nc_handle.close()
        return None
