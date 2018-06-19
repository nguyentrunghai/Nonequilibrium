"""
functions to write namd configuration files
"""


COMMON_NAMD = """

# Force-Field Parameters
exclude             scaled1-4
1-4scaling          1.0
cutoff              12.0
switching           on
switchdist          10.0
pairlistdist        13.5

# Integrator Parameters
timestep            2.0  ;# 2fs/step
rigidBonds          all  ;# needed for 2fs steps
nonbondedFreq       1
fullElectFrequency  2  
stepspercycle       10

# Constant Temperature Control
langevin            on    ;# do langevin dynamics
langevinDamping     1     ;# damping coefficient (gamma) of 1/ps
langevinTemp        $temperature
langevinHydrogen    no    ;# don't couple langevin bath to hydrogens

# Output
restartfreq         5000
dcdfreq             500     ;# 500steps = every 1ps
outputEnergies      500

"""


def write_namd_conf_smd_da(coordinates, structure, parameters, outputName,
                           temperature, tcl_force, steps,
                           out,
                           coordinates_res=None,
                           velocities_res=None):
    """
    :param coordinates: str, initial pdb file name
    :param structure: str, psf file name
    :param parameters: str, charmm prm file name
    :param outputName: str, namd output name
    :param temperature: float, simulation temperature
    :param tcl_force: str, tcl force file name
    :param steps: int, number of namd steps
    :param out: str, name of output file
    :param coordinates_res: str or None, name of coordinates restart file
    :param velocities_res: str or None, name of velocities restart file
    :return: None
    """
    out_string = "\n"
    out_string += "structure       " + structure + "\n"
    out_string += "coordinates     " + coordinates + "\n"
    out_string += "outputName      " + outputName + "\n"
    out_string += "\n"
    out_string += "set temperature %f\n" % temperature
    out_string += "\n"

    out_string += "# Input\n"
    out_string += "paraTypeCharmm    on\n"
    out_string += "parameters       " + parameters + "\n"

    if coordinates_res is not None:
        out_string += "binCoordinates    " + coordinates_res + "\n"

    if velocities_res is not None:
        out_string += "binvelocities     " + velocities_res + "\n"
    else:
        out_string += "temperature  $temperature\n"

    out_string += COMMON_NAMD

    out_string += "# Tcl interface\n"
    out_string += "tclForces        on\n"
    out_string += "tclForcesScript       " + tcl_force + "\n"
    out_string += "\n"

    out_string += "run       %d\n"%steps

    open(out, "w").write(out_string)

    return None


def write_namd_conf_smd_cuc(coordinates, prmtop, host_fixed_pdb,
                            outputName,
                            temperature,
                            tcl_force,
                            steps,
                            out,
                            coordinates_res=None,
                            velocities_res=None):
    """
    :param coordinates: str, initial pdb file name
    :param prmtop: str, amber topology file
    :param host_fixed_pdb: str, pdb file with B column marked to indicate which atoms are fixed
    :param outputName: str, namd output name
    :param temperature: float, simulation temperature
    :param tcl_force: str, tcl force file name
    :param steps: int, number of namd steps
    :param out: str, name of output namd configuration file
    :param coordinates_res: str or None, name of coordinates restart file
    :param velocities_res: str or None, name of velocities restart file
    :return: None
    """
    out_string = "\n"
    out_string += "outputName        %s\n" % outputName
    out_string += "set temperature  %f\n" % temperature
    out_string += "\n"

    out_string += "amber            on\n"
    out_string += "coordinates     " + coordinates + "\n"
    out_string += "parmfile        " + prmtop + "\n"

    if coordinates_res is not None:
        out_string += "binCoordinates    " + coordinates_res + "\n"

    if velocities_res is not None:
        out_string += "binvelocities     " + velocities_res + "\n"
    else:
        out_string += "temperature  $temperature\n"

    out_string += COMMON_NAMD

    out_string += "# fixed the host\n"
    out_string += "fixedAtoms       on\n"
    out_string += "fixedAtomsFile   " + host_fixed_pdb + "\n"
    out_string += "fixedAtomsCol    B\n"
    out_string += "\n"

    out_string += "# Tcl interface\n"
    out_string += "tclForces        on\n"
    out_string += "tclForcesScript       " + tcl_force + "\n"
    out_string += "\n"

    out_string += "run       %d\n" % steps

    open(out, "w").write(out_string)
    return None
