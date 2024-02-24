# /home/lls34/rds/hpc-work/GitHub/CarbonCaptureNMR/systems/02_KHCO3/01-First-Develop/01-Bulk/MACE/hkco3-it0-1e-5-3.model
"""Demonstrates molecular dynamics with constant temperature."""
import argparse
import os
import time
import scipy
import ase.io
import numpy as np
from ase import units
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                         Stationary, ZeroRotation)
from copy import deepcopy
from mace.calculators.mace import MACECalculator
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to XYZ configurations", required=True)
    parser.add_argument(
        "--config_index", help="index of configuration", type=int, default=-1
    )
    parser.add_argument(
        "--error_threshold", help="error threshold", type=float, default=0.1
    )
    parser.add_argument("--temperature_K", help="temperature", type=float, default=300)
    parser.add_argument("--friction", help="friction", type=float, default=0.01)
    parser.add_argument("--timestep", help="timestep", type=float, default=1)
    parser.add_argument("--nsteps", help="number of steps", type=int, default=1000)
    parser.add_argument(
        "--nprint", help="number of steps between prints", type=int, default=10
    )
    parser.add_argument(
        "--nsave", help="number of steps between saves", type=int, default=10
    )
    parser.add_argument(
        "--ncheckerror", help="number of steps between saves", type=int, default=10
    )

    parser.add_argument(
        "--model",
        help="path to model. Use wildcards to add multiple models as committe eg "
        "(`mace_*.model` to load mace_1.model, mace_2.model) ",
        required=True,
    )
    parser.add_argument("--output", help="output path", required=True)
    parser.add_argument(
        "--device",
        help="select device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
    )
    parser.add_argument(
        "--default_dtype",
        help="set default dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument(
        "--compute_stress",
        help="compute stress",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--info_prefix",
        help="prefix for energy, forces and stress keys",
        type=str,
        default="MACE_",
    )
    return parser.parse_args()

def f_unc(dyn, reg=0.2):
    atomsi = dyn.atoms
    force_comm = atomsi.calc.results["forces_comm"]
    force_var= np.var(force_comm, axis=0)
    force = atomsi.get_forces()
    ferr = np.sqrt(np.sum(force_var, axis=1))
    ferr_rel = ferr / (np.linalg.norm(force, axis=1) + reg)
    return ferr_rel, force_var

def printenergy(dyn, start_time=None, reg=0.2):  # store a reference to atoms in the definition.
    """Function to print the potential, kinetic and total energy."""
    a = dyn.atoms
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    ferr_rel, force_var= f_unc(dyn, reg)
    # print(np.max(ferr_rel))
    if start_time is None:
        elapsed_time = 0
    else:
        elapsed_time = time.time() - start_time
    print(elapsed_time,dyn.get_time(),epot,ekin,ekin / (1.5 * units.kB), np.max(ferr_rel))
    # print(
    #     "%.1fs: Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  "  # pylint: disable=C0209
    #     "Etot = %.3feV t=%.1ffs Eerr = %.3feV Ferr = %.3feV/A"
    #     % (
    #         elapsed_time,
    #         epot,
    #         ekin,
    #         ekin / (1.5 * units.kB),
    #         epot + ekin,
    #         dyn.get_time() / units.fs,
    #         a.calc.results["energy_var"],
    #         np.max(np.linalg.norm(a.calc.results["forces_var"], axis=1)),
    #     ),
    #     flush=True,
    # )


def save_config(dyn, fname, threshold, reg=0.2):
    atomsi = dyn.atoms
    ens = atomsi.get_potential_energy()
    ferr_rel, force_var= f_unc(dyn, reg)
    atomsi.info.update(
        {
            "mlff_energy": ens,
            "time": np.round(dyn.get_time() / units.fs, 5),
            "mlff_energy_var": atomsi.calc.results["energy_var"],
            "max_ferr_rel": np.max(ferr_rel),
        }
    )
    atomsi.arrays.update(
        {"mlff_forces_var": force_var,
        "rel_force_uncer": ferr_rel}
    )

    ase.io.write(fname, atomsi, append=True)
    if np.max(ferr_rel) > threshold:
        print(
            "Error too large {:.3}. Stopping t={:.2} fs.".format(  # pylint: disable=C0209
                np.max(ferr_rel), dyn.get_time() / units.fs
            ),
            flush=True,
        )
        dyn.max_steps = 0
        sys.exit(0)


# def stop_error(dyn, threshold, reg=0.2):
#     atomsi = dyn.atoms
#     ferr_rel, force_var= f_unc(dyn, reg)
#     # print(np.max(ferr_rel))



def main():
    args = parse_args()

    mace_fname = args.model
    atoms_fname = args.config
    atoms_index = args.config_index
    calc = MACECalculator(
        mace_fname,
        args.device,
        default_dtype=args.default_dtype,
    )
    # mace_calc = MACECalculator(
    #     mace_fname,
    #     args.device,
    #     default_dtype=args.default_dtype,
    # )

    NSTEPS = args.nsteps

    if os.path.exists(args.output):
        print("Trajectory exists. Continuing from last step.")
        atoms = ase.io.read(args.output, index=-1)
        len_save = len(ase.io.read(args.output, ":"))
        print("Last step: ", atoms.info["time"], "Number of configs: ", len_save)
        NSTEPS -= len_save * args.nsave
    else:
        atoms = ase.io.read(atoms_fname, index=atoms_index)
        # MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_K)

    atoms.calc = calc

    # We want to run MD with constant energy using the Langevin algorithm
    # with a time step of 5 fs, the temperature T and the friction
    # coefficient to 0.02 atomic units.
    # dyn = Langevin(
    #     atoms=atoms,
    #     timestep=args.timestep * units.fs,
    #     temperature_K=args.temperature_K,
    #     friction=args.friction,
    # )
    def rot_upper_tx(cell):
        R, Q = scipy.linalg.rq(cell)
        return ase.cell.Cell(R), Q
    def rotate_system(ats):
        U, R = rot_upper_tx(ats.get_cell())
        newcell = ats.get_cell() @ R.transpose()
        newcell[1,0] = 0.0
        newcell[2,0] = 0.0
        newcell[2,1] = 0.0
        ats.set_cell(newcell)
        ats.set_positions(ats.get_positions() @ R.transpose())

    rotate_system(atoms)
    def flip_system(ats):
        oldcell = deepcopy(ats.get_cell())
        oldpos = deepcopy(ats.get_positions())
        newcell = - oldcell
        newpos = - oldpos
        ats.set_cell(newcell)
        ats.set_positions(newpos)

    if np.linalg.det(atoms.get_cell())<0:
        print(np.linalg.det(atoms.get_cell()))
        flip_system(atoms)
    dyn = NPT(
        atoms, 
        1.0*units.fs, 
        temperature_K=args.temperature_K, 
        externalstress=0.0, 
        ttime=100*units.fs, 
        pfactor=1000*units.fs, 
        trajectory=f"./{args.config_index}.traj", 
        logfile=f"./{args.config_index}.log",
        loginterval=10)
    
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_K)
    Stationary(atoms)
    ZeroRotation(atoms)
    dyn.attach(printenergy, interval=args.nsave, dyn=dyn, start_time=time.time())
    dyn.attach(save_config, interval=args.nsave, dyn=dyn, fname=args.output, threshold=args.error_threshold)
    # dyn.attach(
    #     stop_error, interval=args.ncheckerror, dyn=dyn, threshold=args.error_threshold
    # )
    # Now run the dynamics
    dyn.run(0)
    dyn.run(NSTEPS)


if __name__ == "__main__":
    main()
