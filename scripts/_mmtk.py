
from __future__ import print_function
import os

from MMTK import InfiniteUniverse
from MMTK.PDB import PDBConfiguration
from MMTK.ForceFields import Amber12SBForceField
from Scientific.Geometry import Vector
from MMTK.ForceFields.Restraints import HarmonicDistanceRestraint, HarmonicTrapForceField
from MMTK.Collections import Collection
from MMTK import Units


import MMTK.Biopolymers
MMTK.Biopolymers.defineAminoAcidResidue("alaninen", code3="ALN", code1=None)
MMTK.Biopolymers.defineAminoAcidResidue("alaninec", code3="ALC", code1=None)

from MMTK.PDB import defineMolecule
defineMolecule('CUC', 'cuc')
defineMolecule('B5', 'b5')
defineMolecule('FB', 'fb')

GAFF_DIR = os.environ["MMTKGAFF"]
AMBER_DIR = os.environ["AMBERFF"]

GAFF_PARAMETER_FILE = os.path.join(GAFF_DIR, "gaff.dat")
GAFF_MOD_FILES = [os.path.join(GAFF_DIR, "cb7_am1-bcc.frcmod"), os.path.join(GAFF_DIR, "hexafluorobenzene.frcmod") ]

AMBER_PARAMETER_FILE = os.path.join(AMBER_DIR, "parm10.dat")
AMBER_MOD_FILES = [os.path.join(AMBER_DIR, "frcmod.ff12SB")]

ALLOWED_FF = ["Amber12SB", "Gaff"]


def _get_Amber12SB_ff():
    """
    return MMTK.ForceFields.Amber.AmberForceField.Amber12SBForceField
    """
    return Amber12SBForceField(parameter_file=AMBER_PARAMETER_FILE, mod_files=AMBER_MOD_FILES)


def _get_gaff_ff():
    """
    return MMTK.ForceFields.Amber.AmberForceField.Amber12SBForceField
    """
    return Amber12SBForceField(parameter_file=GAFF_PARAMETER_FILE, mod_files=GAFF_MOD_FILES)


def get_ff(name):
    """
    name    :   str
    """
    assert name in ALLOWED_FF, "unknown ff " + name
    if name == "Amber12SB":
        return _get_Amber12SB_ff()
    elif name == "Gaff":
        return _get_gaff_ff()


def _get_molecules(pdb_file):
    configuration = PDBConfiguration(pdb_file)
    molecules = configuration.createAll()
    return molecules


def start_universe(pdb_file):
    """
    pdb_file:   str,    pdb file name
    return  MMTK.Universe
    """
    universe = InfiniteUniverse()
    molecules = _get_molecules(pdb_file)
    universe.addObject(molecules)
    return universe


class end_2_end_harmonic_restraint(object):
    def __init__(self, universe, k):
        """
        universe    :   MMTK.Universe.InfiniteUniverse
        k           :   force constant, unit in ( kj/mol ) / nm^2
        """
        self._two_end_atoms = self._get_two_ends_of_deca_alanine(universe)
        self._k = k * Units.kJ / Units.mol / Units.nm**2
        self._initial_l = self._cal_initial_lambda(universe)
        self._print_info()

    def _get_two_ends_of_deca_alanine(self, universe):
        """
        return [start_atom, end_atom]   :   list of MMTK.ChemicalObjects.Atom objects
        """
        start_atom = universe[0][0][0].peptide.N
        end_atom = universe[0][0][-1].peptide.N1
        return [start_atom, end_atom]

    def _cal_initial_lambda(self, universe):
        """
        lambda at t=0
        """
        return universe.distance(self._two_end_atoms[0], self._two_end_atoms[1])

    def _print_info(self):
        print("\nRestraint information")
        print("k = ", self._k)
        print("between two atoms")
        for a in self._two_end_atoms:
            print(a, ", index = ", a.index)
        print(" ")
        return None

    def get_ff(self, l):
        """
        l             :   reference distance, unit in nm
        """
        return HarmonicDistanceRestraint(self._two_end_atoms[0], self._two_end_atoms[1], l*Units.nm, self._k)

    def get_initial_l(self):
        return self._initial_l

    def get_restrained_atoms(self):
        return self._two_end_atoms


class restraint_atoms_to_positions(object):
    def __init__(self, universe, restrained_molecule_index, ref_molecule_pos_pdb, k, carbons_only):
        """
        universe    :   MMTK.Universe.InfiniteUniverse, universe of the complex, or the molecule alone
        restrained_molecule_index  :   int, index of molecule to be restrained in universe
        ref_molecule_pos_pdb:  str,    file name of the pdb containing only the molecule to be restrained
        k           :   float, force constant, unit in ( kj/mol ) / nm^2
        carbons_only    :   bool, only restrain C if True, otherwise, all heavy atoms will be restrained
        ref_host_pdb    :   str,    file name of the pdb containing only the host
        """
        assert restrained_molecule_index < len(universe), "host_index is larger than the number of molecules in universe"
        self._universe = universe
        self._restrained_molecule_index = restrained_molecule_index
        self._k = k * Units.kJ / Units.mol / Units.nm**2

        self._ref_universe = start_universe(ref_molecule_pos_pdb)

        self._restrained_atoms = self._get_restrained_atoms(carbons_only)
        self._reference_atoms = self._get_refence_atoms(carbons_only)
        self._check_consistency()
        self._reference_positions = self._get_reference_positions()

        self._print_info()

        self._ff = self._get_restraint_ff()

    def _check_consistency(self):
        restrained_molecule = self._universe[self._restrained_molecule_index]
        ref_moleule         = self._ref_universe[0]
        for a, b in zip(restrained_molecule.atomList(), ref_moleule.atomList()):
            if repr(a) != repr(b):
                raise RuntimeError("No consistency between host in universe and ref")

        for a, b in zip(self._restrained_atoms, self._reference_atoms):
            if repr(a) != repr(b):
                raise RuntimeError("No consistency between self._restrained_atoms and self._reference_atoms")
        return None

    def _get_heavy_atoms(self, universe, molecule_index):
        molecule = universe[molecule_index]
        sel_atoms = [atom for atom in molecule.atomList() if atom.type.name != "hydrogen"]
        return sel_atoms

    def _get_carbons(self, universe, molecule_index):
        molecule = universe[molecule_index]
        sel_atoms = [atom for atom in molecule.atomList() if atom.type.name == "carbon"]
        return sel_atoms

    def _get_restrained_atoms(self, carbons_only):
        if carbons_only:
            return self._get_carbons(self._universe, self._restrained_molecule_index)
        else:
            return self._get_heavy_atoms(self._universe, self._restrained_molecule_index)

    def _get_refence_atoms(self, carbons_only):
        if carbons_only:
            return self._get_carbons(self._ref_universe, 0)
        else:
            return self._get_heavy_atoms(self._ref_universe, 0)

    def _get_reference_positions(self):
        postions = [atom.position() for atom in self._reference_atoms]
        return postions

    def _print_info(self):
        print("\nRestraint information")
        print("k = ", self._k)
        for atom, postion in zip(self._restrained_atoms, self._reference_positions):
            print(atom, " restrained to ", postion)
        print(" ")
        return None

    def _get_restraint_ff(self):
        for i, atom, postion in zip(range(len(self._restrained_atoms)), self._restrained_atoms, self._reference_positions):

            if i == 0:
                ff = HarmonicTrapForceField(atom, postion, self._k)
            else:
                ff = ff + HarmonicTrapForceField(atom, postion, self._k)
        return ff

    def get_ff(self):
        return self._ff


class com_2_x_0_0_harmonic_restraint(object):
    def __init__(self, universe, restrained_molecule_index, k):
        """
        universe    :   MMTK.Universe.InfiniteUniverse, universe of the complex, or the molecule alone
        restrained_molecule_index  :   int, index of molecule to be restrained in universe
        k           :   float, force constant, unit in ( kj/mol ) / nm^2
        """
        assert restrained_molecule_index < len(universe), "host_index is larger than the number of molecules in universe"
        self._k = k * Units.kJ / Units.mol / Units.nm**2
        self._restrained_atoms = self._get_heavy_atoms(universe, restrained_molecule_index)
        self._initial_l = self._cal_initial_lambda()
        self._print_info()

    def _get_heavy_atoms(self, universe, restrained_molecule_index):
        molecule = universe[restrained_molecule_index]
        col = Collection()
        for atom in molecule.atomList():
            if atom.type.name != "hydrogen":
                col.addObject(atom)
        return col

    def _cal_initial_lambda(self):
        com = self._restrained_atoms.centerOfMass()
        return com.array[0]

    def _print_info(self):
        print("\nRestraint information")
        print("\nCOM of the following atoms is restrained")
        print("k = ", self._k)
        for a in self._restrained_atoms:
            print(a, ", index = ", a.index)
        print(" ")
        return None

    def get_ff(self, lx):
        """
        lx in nm
        """
        postion = Vector(lx*Units.nm, 0., 0.)
        return HarmonicTrapForceField(self._restrained_atoms, postion, self._k)

    def get_initial_l(self):
        return self._initial_l

    def get_restrained_atoms(self):
        return self._restrained_atoms
