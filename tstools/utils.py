import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw, rdChemReactions


def get_fragments(molobj):
    return list(Chem.GetMolFrags(molobj, asMols=True, sanitizeFrags=False))


def get_ac(molobj):
    """ """
    return Chem.GetAdjacencyMatrix(molobj)


def get_num_bonds(molobj):
    """ """
    return molobj.GetNumBonds()


def molobj_axyz(molobj, confid=-1):
    """ """
    atom_symbols = [atom.GetSymbol() for atom in molobj.GetAtoms()]
    xyz = molobj.GetConformer(id=confid).GetPositions()
    return atom_symbols, xyz


def get_bond_change_idx(reactant, product):
    """ """
    ac_reactant = get_ac(reactant)
    ac_product = get_ac(product)

    bond_change = ac_reactant - ac_product
    bond_change_idx = np.where(np.triu(bond_change) != 0)
    return list(zip(*bond_change_idx))


def draw_reaction(
    reactants: list[Chem.Mol], products: list[Chem.Mol], filename: str = "reaction.png"
) -> None:
    """ """

    rxn = rdChemReactions.ChemicalReaction()
    for reactant in reactants:
        rxn.AddReactantTemplate(reactant)

    for product in products:
        rxn.AddProductTemplate(product)

    rxn_png = Draw.ReactionToImage(rxn, returnPNG=True, useSVG=True)
    with open(filename, "w") as pngfile:
        pngfile.write(rxn_png)
