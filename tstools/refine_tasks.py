import logging

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers, rdMolAlign
from rdkit.ForceField import rdForceField

_logger = logging.getLogger()


def interpolate_structures(
    path_points: list[np.ndarray], atom_symbols, n_points = 20, calculator = None
):
    """
    """
    if len(path_points) != 2:
        raise RuntimeError('Can only interpolate between two structures')

    path_m1, path_p1 = path_points
    n_atoms = len(atom_symbols)
    
    difference_mat = path_m1 - path_p1
    interpolated_coords = np.zeros((n_points, n_atoms, 3))
    interpolated_energies = np.zeros(n_atoms)
    for i in range(n_points + 1):
        interpolated_coords[i - 1] = path_p1 + i / n_points * difference_mat

    if calculator is None:
        return interpolated_coords, interpolated_energies

    for i, coords in enumerate(interpolated_coords):
        energies = calculator(atom_symbols, coords, namespace="interpolate")['energy']
        interpolated_energies[i] = energies['elec_energy']
    
    return interpolated_coords, interpolated_energies


def refine_uff_dist_constrained(
    molobj,
    bond_idx,
    min_distance=4.0,
    max_distance=6.0,
    use_vdW=False,  # TODO: Set vdw
    force_constant=1.0,
    opt_max_iters=1_000,
    opt_force_tol=1e-4,
    opt_energy_tol=1e-6,
):
    """
    Refines embedded molobj by performing a constrained UFF optimization.
    
    The distance constraints between `bond_idx` makes sure that 
    the fragments are but in somewhat the correct position.
    """
    if not rdForceFieldHelpers.UFFHasAllMoleculeParams(molobj):
        raise NotImplementedError("UFF doesn't have parameters!")

    Chem.SanitizeMol(molobj)

    ff = rdForceFieldHelpers.UFFGetMoleculeForceField(
        molobj, ignoreInterfragInteractions=False
    )
    for atom_i, atom_j in bond_idx:
        ff.UFFAddDistanceConstraint(
            int(atom_i),
            int(atom_j),
            False,
            float(min_distance),
            float(max_distance),
            float(force_constant),
        )
    ff.Initialize()

    exit_msg = ff.Minimize(
        maxIts=opt_max_iters, forceTol=opt_force_tol, energyTol=opt_energy_tol
    )
    if exit_msg != 0:
        _logger.warning("Constrained FF optimization didn't converge.")

    return molobj


def refine_uff_constrained_embed(reactant, product, force_constant=100.0):
    """ 
    Refines the reactant or product similarly to RDKit's contained embed.
    Tries to fit `mol` to the core through a heavily constrained FF optimization.

    The core is the molecule (reactant or product) with the most bonds.
    """

    if not rdForceFieldHelpers.UFFHasAllMoleculeParams(reactant):
        raise NotImplementedError("UFF doesn't have reactant parameters!")

    if not rdForceFieldHelpers.UFFHasAllMoleculeParams(product):
        raise NotImplementedError("UFF doesn't have product parameters!")

    n_bonds_reactant = reactant.GetNumBonds()
    n_bonds_product = product.GetNumBonds()

    if n_bonds_reactant >= n_bonds_product:
        reac_is_core = True
        mol, core = product, reactant
    else:
        reac_is_core = False
        mol, core = reactant, product

    coreConf = core.GetConformer()
    ff = rdForceFieldHelpers.UFFGetMoleculeForceField(
        mol, ignoreInterfragInteractions=False
    )
    for i in range(core.GetNumAtoms()):
        for j in range(i + 1, core.GetNumAtoms()):
            corePtI = coreConf.GetAtomPosition(i)
            corePtJ = coreConf.GetAtomPosition(j)

            d = corePtI.Distance(corePtJ)
            ff.AddDistanceConstraint(i, j, d, d, force_constant)

    ff.Initialize()
    n = 4
    more = ff.Minimize()
    while more and n:
        ff.Minimize()
        n -= 1

    if reac_is_core:
        return core, mol
    else:
        return mol, core
