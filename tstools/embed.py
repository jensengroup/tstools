import logging
import copy
from itertools import product

import numpy as np

from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    rdDistGeom,
    rdMolTransforms,
    rdmolops,
)
from rdkit.Geometry import Point3D

from tstools.utils import get_fragments

_logger = logging.getLogger("embed")


def reassign_atom_idx(mol):
    """Reassigns the RDKit mol object atomid to atom mapped id"""
    renumber = [(atom.GetIdx(), atom.GetAtomMapNum()) for atom in mol.GetAtoms()]
    new_idx = [idx[0] for idx in sorted(renumber, key=lambda x: x[1])]
    mol = Chem.RenumberAtoms(mol, new_idx)
    rdmolops.AssignStereochemistry(mol, force=True)
    return mol


class EmbedError(Exception):
    pass


def embed_fragment(molobj: Chem.Mol, seed: int = 31) -> Chem.Mol:
    """Embed fragment/simple molecule"""
    rdmol = copy.deepcopy(molobj)
    try:
        rdDistGeom.EmbedMolecule(
            rdmol,
            useRandomCoords=False,
            randomSeed=seed,
            maxAttempts=10_000,
            ETversion=1,
        )
    except:
        raise EmbedError("RDKit Failed to embed Molecule.")

    Chem.SanitizeMol(rdmol)
    AllChem.UFFOptimizeMolecule(rdmol)

    return rdmol


def embed_fragments(molobjs: list[Chem.Mol], seed: int = 31) -> list[Chem.Mol]:
    """Embed list of mol objs"""
    embedded_molobjs = []
    for molobj in molobjs:
        embedded_molobjs.append(embed_fragment(molobj, seed=seed))
    return embedded_molobjs


def _sort_fragments_size(fragments):
    """ """
    frag_size = [frag.GetNumAtoms() for frag in fragments]
    sorted_frags = sorted(
        list(zip(fragments, frag_size)), key=lambda x: x[1], reverse=True
    )
    return [x[0] for x in sorted_frags]


def center_fragments(molobjs: list[Chem.Mol]) -> None:
    """ """
    for molobj in molobjs:
        confobj = molobj.GetConformer()
        centroid_point = rdMolTransforms.ComputeCentroid(confobj)

        for i in range(molobj.GetNumAtoms()):
            orginal_point = confobj.GetAtomPosition(i)
            confobj.SetAtomPosition(i, orginal_point - centroid_point)


def _translate_fragment(molobj: Chem.Mol, direction: list[float]):
    """ """
    confobj = molobj.GetConformer()
    for i in range(confobj.GetNumAtoms()):
        atom_pos = confobj.GetAtomPosition(i)
        atom_pos += direction
        confobj.SetAtomPosition(i, atom_pos)
    return molobj


def _check_embedded_fragments(new_molobj, old_molobj, cutoff=1.5):
    """
    Check if atoms of two molecules are too close.
    """
    new_fragment = new_molobj.GetConformer().GetPositions()
    old_fragment = old_molobj.GetConformer().GetPositions()

    for i, j in product(range(old_fragment.shape[0]), range(new_fragment.shape[0])):
        if i >= j:
            if np.linalg.norm(new_fragment[j] - old_fragment[i]) < cutoff:
                return False
    return True


def simple_embed_and_translate(
    molobj: Chem.Mol, seed: int = 31, direction: list[float] = [0.4, 0.0, 0.0]
):
    """
    Embed fragments, and move fragment in `direction`.
    If more than 2 fragments it just embed on a line.
    """

    molobjs = get_fragments(molobj)
    frag_molobjs = embed_fragments(molobjs, seed=seed)
    center_fragments(frag_molobjs)
    max_fragment_size = frag_molobjs[0].GetNumAtoms()

    direction = Point3D(*direction)
    for fragid, fragment_molobj in enumerate(frag_molobjs):
        if fragid == 0:
            merged_molobj = copy.deepcopy(fragment_molobj)
            continue

        fragment_molobj = _translate_fragment(
            fragment_molobj, direction * max_fragment_size
        )

        merged_molobj = Chem.CombineMols(merged_molobj, fragment_molobj)
        max_fragment_size += fragment_molobj.GetNumAtoms()

    return reassign_atom_idx(merged_molobj)


def random_embedding(
    molobj: Chem.Mol,
    seed: int = 42,
    translation_distance: float = 4.0,
    distance_cutoff=2.0,
    max_attempts=10,
):
    """Embed similarly to ChemDyME:
    R. J. Shannon, E. M. Nunez, D. V. Shalashilin, D. R. Glowacki,
    arXiv [physics.chem-ph] 2021. http://arxiv.org/abs/2104.02389
    """
    np.random.seed(seed=seed)

    molobjs = get_fragments(molobj)
    frag_molobjs = embed_fragments(molobjs, seed=seed)
    center_fragments(frag_molobjs)

    for fragid, fragment_molobj in enumerate(frag_molobjs):
        if fragid == 0:
            merged_molobj = copy.deepcopy(fragment_molobj)
            continue

        # Keep making a random embedding until fragments are
        # far enough way from eavh other.
        count = 0
        while (
            _check_embedded_fragments(
                fragment_molobj, merged_molobj, cutoff=distance_cutoff
            )
            is False
        ):
            random_vector = np.random.rand(3)
            unit_vector = random_vector / np.linalg.norm(random_vector)
            fragment_molobj = _translate_fragment(
                fragment_molobj, Point3D(*(translation_distance * unit_vector))
            )
            count += 1

            if count == max_attempts:
                _logger.critical("Can't embbed molecule")
                return None

        merged_molobj = Chem.CombineMols(merged_molobj, fragment_molobj)

    return reassign_atom_idx(merged_molobj)
