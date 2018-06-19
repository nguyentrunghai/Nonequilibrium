"""
define functions that write namd file for runing umbrella sampling
"""
from __future__ import print_function

import numpy as np
import pandas as pd


COMMON_TCL_DA = """
# pulling velocity (A/timestep)
set v 0.0

proc calcforces {} {

  global Tclfreq t k v a1 a2 c1x c1y c1z c2x c2y c2z outfilename
  

  # get coordinates

  loadcoords coordinate

  set r1 $coordinate($a1)
  set r1x [lindex $r1 0]
  set r1y [lindex $r1 1]
  set r1z [lindex $r1 2]

  set r2 $coordinate($a2)
  set r2x [lindex $r2 0]
  set r2y [lindex $r2 1]
  set r2z [lindex $r2 2]

  # calculate forces

  set f1x [expr $k*($c1x-$r1x)]
  set f1y [expr $k*($c1y-$r1y)]
  set f1z [expr $k*($c1z-$r1z)]
  lappend f1 $f1x $f1y $f1z

  set f2x [expr $k*($c2x-$r2x)]
  set f2y [expr $k*($c2y-$r2y)]
  #set f2z [expr $k*($c2z+$v*$t-$r2z)]
  set f2z [expr $k*($c2z-$r2z)]
  lappend f2 $f2x $f2y $f2z

  # apply forces

  addforce $a1 $f1
  addforce $a2 $f2

  # output

  set foo [expr $t % $Tclfreq]
  if { $foo == 0 } {
      set outfile [open $outfilename a]
      set time [expr $t*2/1000.0]
      puts $outfile "$time $r2z $f2z"
      close $outfile
  }
  incr t
  return
}

"""


def write_tcl_us_da(out, c2z, force_constant=7.2, force_out="tclforce.out"):
    """
    :param out: str, output file name
    :param c2z: float, window center
    :param force_constant: float, harmonic force constant, (kcal/mol/A^2)
    :param force_out: str, file name to output restraint coordinate and force
    :return: None
    """
    out_string = "# $Id: smd.tcl,v 1.2 2005/02/18 18:07:11 mbach Exp $\n"
    out_string += "\n"

    out_string += "# Atoms selected for force application\n"
    out_string += "\n"

    out_string += "set id1 [atomid BH 1 N]\n"
    out_string += "set grp1 {}\n"
    out_string += "lappend grp1 $id1\n"
    out_string += "set a1 [addgroup $grp1]\n"
    out_string += "\n"

    out_string += "set id2 [atomid BH 10 NT]\n"
    out_string += "set grp2 {}\n"
    out_string += "lappend grp2 $id2\n"
    out_string += "set a2 [addgroup $grp2]\n"
    out_string += "\n"

    out_string += "# set the output frequency, initialize the time counter\n"
    out_string += "set Tclfreq 500\n"
    out_string += "set t 0\n"
    out_string += "\n"

    out_string += "# contraint points\n"
    out_string += "\n"

    out_string += "set c1x 0.0\n"
    out_string += "set c1y 0.0\n"
    out_string += "set c1z 0.0\n"
    out_string += "\n"

    out_string += "set c2x 0.0\n"
    out_string += "set c2y 0.0\n"
    out_string += "set c2z %f\n" % c2z
    out_string += "\n"

    out_string += "# force constant (kcal/mol/A^2)\n"
    out_string += "set k %f\n" % force_constant
    out_string += "\n"

    out_string += "set outfilename " + force_out + "\n"
    out_string += "open $outfilename w\n"
    out_string += "\n"

    out_string += COMMON_TCL_DA

    open(out, "w").write(out_string)
    return None


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
restartfreq         50000
dcdfreq             500     ;# 500steps = every 1ps
outputEnergies      500

"""


def write_namd_conf_us_da(out,
                          coordinates, structure, parameters,
                          outputName,
                          temperature,
                          tcl_force,
                          min_steps,
                          steps,
                          coordinates_res=None,
                          velocities_res=None):
    """
    :param out: str, name of output namd configuration file
    :param coordinates: str, initial pdb file name
    :param structure: str, psf file name
    :param parameters: str, charmm prm file name
    :param outputName: str, namd output name
    :param temperature: float, simulation temperature
    :param tcl_force: str, tcl force file name
    :param min_steps: int, number of minimization steps
    :param steps: int, number of equilibarte and production steps
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

    out_string += "minimize %d\n" % min_steps
    out_string += "\n"
    out_string += "run       %d\n" % steps

    open(out, "w").write(out_string)

    return None


COMMON_TCL_CUC = """

# Atoms selected for force application 
set grp {}
lappend grp 2
lappend grp 3
lappend grp 5
lappend grp 7 
lappend grp 9 
lappend grp 11
set at [addgroup $grp]

# set the output frequency, initialize the time counter
set Tclfreq 500
set t 0

# pulling velocity (A/timestep)
set v 0.0

proc calcforces {} {

  global Tclfreq t k v at cx cy cz outfilename

  # get coordinates

  loadcoords coordinate

  set r $coordinate($at)
  set rx [lindex $r 0]
  set ry [lindex $r 1]
  set rz [lindex $r 2]

  # calculate forces

  #set fx [expr $k*($cx+$v*$t-$rx)]
  set fx [expr $k*($cx-$rx)]
  set fy [expr $k*($cy-$ry)]
  set fz [expr $k*($cz-$rz)]
  lappend f $fx $fy $fz

  # apply forces

  addforce $at $f

  # output

  set foo [expr $t % $Tclfreq]
  if { $foo == 0 } {
      set outfile [open $outfilename a]
      set time [expr $t*2/1000.0]
      puts $outfile "$time $rx $fx"
      close $outfile
  }
  incr t
  return
}

"""


def write_tcl_us_cuc(out, cx, force_constant=7.2, force_out="tclforce.out"):
    """
    :param out:  str, name of output namd configuration file
    :param cx: float, window center
    :param force_constant: float, harmonic force constant, (kcal/mol/A^2)
    :param force_out:  str, file name to output restraint coordinate and force
    :return: None
    """
    out_string = "# $Id: smd.tcl,v 1.2 2005/02/18 18:07:11 mbach Exp $\n"
    out_string += "\n"

    out_string += "# contraint points\n"
    out_string += "set cx   %f\n" % cx
    out_string += "set cy    0.0\n"
    out_string += "set cz    0.0\n"
    out_string += "\n"

    out_string += "# force constant (kcal/mol/A^2)\n"
    out_string += "set k %f\n" % force_constant
    out_string += "\n"

    out_string += "set outfilename " + force_out + "\n"
    out_string += "open $outfilename w\n"
    out_string += "\n"

    out_string += COMMON_TCL_CUC

    open(out, "w").write(out_string)
    return None


def write_namd_conf_us_cuc(out,
                            coordinates, prmtop, host_fixed_pdb,
                            outputName,
                            temperature,
                            tcl_force,
                            min_steps,
                            steps,
                            coordinates_res=None,
                            velocities_res=None):
    """
    :param out: str, name of output namd configuration file
    :param coordinates: str, initial pdb file name
    :param prmtop: str, amber topology file
    :param host_fixed_pdb: str, pdb file with B column marked to indicate which atoms are fixed
    :param outputName: str, namd output name
    :param temperature: float, simulation temperature
    :param tcl_force: str, tcl force file name
    :param min_steps: int, number of minimization steps
    :param steps: int, number of namd steps
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

    out_string += "minimize %d\n" % min_steps
    out_string += "\n"

    out_string += "run       %d\n" % steps

    open(out, "w").write(out_string)
    return None


def load_namd_energy(file_name):
    """
    :param file_name: str
    :return: energies, pandas DataFrame
    """
    energy_lines = []
    header_line = None
    with open(file_name, "r") as handle:
        for line in handle:
            if line.startswith("ETITLE:"):
                header_line = line

            elif line.startswith("ENERGY:"):
                energy_lines.append(line)

    if header_line is None:
        raise Exception("Cannot find header line")
    if len(energy_lines) == 0:
        raise Exception("There is no energy line")

    columns = header_line.split()[1:]
    ncols = len(columns)
    nlines = len(energy_lines)

    data = np.zeros([nlines, ncols])

    for i, line in enumerate(energy_lines):
        data[i,:] = np.array( [ float(entry) for entry in line.split()[1:] ] )

    return pd.DataFrame(data, columns=columns)


def load_namd_potential(file_name):
    """
    :param file_name: str
    :return: ndarray
    """
    energy = load_namd_energy(file_name)
    return np.array(energy["POTENTIAL"])


def load_biased_coordinates(file_name):
    """
    :param file_name: str
    :return: ndarray
    """
    data = np.loadtxt(file_name)
    return data[:, 1]


def harmonic_potential_cal(force_constant, us_center, coordinates):
    """
    :param force_constant: float
    :param us_center: float
    :param coordinates: float or ndarray
    :return: float or ndarray
    """
    assert isinstance(force_constant, float), "force_constant must be float"
    assert isinstance(us_center, float), "us_center must be float"
    pot_e = 0.5 * force_constant * (coordinates - us_center) ** 2
    return pot_e


def cal_u_kln(force_constant, us_centers, coordinates, unbiased_potentials):
    """
    :param force_constant: float
    :param us_centers: 1d array
    :param coordinates: 2d array
    :param unbiased_potentials: 2d array
    :return: u_kln
    """
    assert us_centers.ndim == 1, "us_centers must be 1d array"
    assert coordinates.ndim == unbiased_potentials.ndim == 2, "coordinates and unbiased_potentials must be 2d array"
    assert coordinates.shape == unbiased_potentials.shape, "coordinates and unbiased_potentials must have same shape"
    assert us_centers.shape[0] == coordinates.shape[0], "us_centers and coordinates must have the same K"

    K = us_centers.shape[0]
    N = coordinates.shape[1]
    print("K = ", K, "N = ", N)

    u_kln = np.zeros((K, K, N), dtype=float)

    for k in range(K):
        coord_k = coordinates[k, :]
        unbiased_pot_k = unbiased_potentials[k]

        for l in range(K):
            center_l = us_centers[l]

            u_kln[k, l, :] = unbiased_pot_k + harmonic_potential_cal(force_constant, center_l, coord_k)

    return u_kln

