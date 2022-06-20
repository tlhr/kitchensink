#!/usr/bin/env python3
"""fftk.py - Tool to convert FFTK output to GROMACS."""

import argparse
from pathlib import Path
from typing import Union, Sequence

import numpy as np
import parmed as pm
from openbabel import pybel as pb

from kitchensink.io import handle_path


def fftk_to_gmx(pdb: Union[str, Path], psf: Union[str, Path],
                par: Sequence[Union[str, Path]], filename: Union[str, Path]="ligand"):
    """Converts the output of FFTK to Gromacs format.

    Parameters
    ----------
    pdb
        Optimized PDB structure file created during FFTK GeoOpt.
    psf
        Charmm PSF file with optimized charges from ChargeOpt.
    par
        List of CHARMM parameter files containing determined parameters.
        Order is important, parameters from earlier files will be overwritten
        by later ones. Recommended order: Force field RTF and PAR, PAR
        with all parameters, PAR with optimized parameters.
    filename
        Output filenames and/or path.

    """

    # We deal with all the IO first
    pdb = handle_path(pdb).as_posix()
    psf = handle_path(psf).as_posix()
    par = [handle_path(f).as_posix() for f in par]
    path = handle_path(filename, non_existent=True)

    struc = pm.charmm.CharmmPsfFile(psf)
    params = pm.charmm.CharmmParameterSet(*par)
    struc.load_parameters(params)

    # PSF files do not contain any coordinate information
    coords = next(pb.readfile(format="pdb", filename=pdb))
    struc.coordinates = np.asarray([at.coords for at in coords.atoms])
    top = pm.gromacs.GromacsTopologyFile.from_structure(struc)

    # For some stupid reason, CHARMM writes the negative of the LJ epsilon
    # by default, which means we need to flip the sign.
    for atom in top.atoms:
        atom.epsilon = abs(atom.epsilon)

    # ITP contains charges, topology, PRM contains parameters, GRO contains coordinates
    top.write(path.with_suffix(".itp").as_posix(), parameters=path.with_suffix(".prm").as_posix())
    top.write(path.with_suffix(".itp").as_posix(), itp=True)
    top.save(path.with_suffix(".gro").as_posix())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", "-s", type=str, help="Optimized PDB structure file created during FFTK GeoOpt")
    parser.add_argument("--psf", "-p", type=str, help="Charmm PSF file with optimized charges from ChargeOpt")
    parser.add_argument("--output", "-o", type=str, help="Base filename for output .itp, .gro and .prm files")
    parser.add_argument("--par", "-r", type=str, nargs='+', help="""
        List of CHARMM parameter files containing determined parameters.
        Order is important, parameters from earlier files will be overwritten
        by later ones. Recommended order: Force field RTF and PAR, PAR
        with all parameters, PAR with optimized parameters.
    """)
    args = parser.parse_args()

    fftk_to_gmx(pdb=args.pdb, psf=args.psf, par=args.par, filename=args.output)
