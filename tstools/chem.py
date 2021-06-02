import numpy as np

from rdkit import Chem
from tstools import embed

class PathMolecule:
    def __init__(self, molblock):
        self._molblock = molblock

    @classmethod
    def from_molfile(cls, filename):
        with open(filename, 'r') as molfile:
            return cls(molfile.read())
    
    @classmethod
    def from_xyz(xyzblock):
        """ """
        pass

    @classmethod
    def from_xyzfile(file):
        pass

    @property
    def rdmol(self):
        rdmol = Chem.MolFromMolBlock(self._molblock, sanitize=False)
        Chem.SanitizeMol(rdmol)
        return rdmol
    
    @property
    def is_3D(self):
        for coord in self.coordinates.T:
            if np.all(coord == coord[0]):
                return False
        return True            

    @property
    def atom_symbols(self):
        return [atom.GetSymbol() for atom in self.rdmol.GetAtoms()]
    
    @property
    def coordinates(self):
        natoms = len(self.atom_symbols)
        info = self._molblock.split("\n")[4 : 4 + natoms]
        coords = np.array([coord.split()[:3] for coord in info], dtype=float)
        return coords
    
    # @coordinates.setter
    # def _update_molblock_coords(self, new_coords):
    #     """
    #     Given an array of coordinates, update the coords of the molblock.
    #     """
    #     tmp_mol = Chem.MolFromMolBlock(self.molblock, sanitize=False)
    #     conf = tmp_mol.GetConformer()
    #     for i in range(tmp_mol.GetNumAtoms()):
    #         x, y, z = new_coords[i]
    #         conf.SetAtomPosition(i, Point3D(x, y, z))

    #     self._molblock = Chem.MolToMolBlock(tmp_mol)
    
    @property
    def ac_matrix(self):
        """ Return the adjecency matrix """
        return Chem.GetAdjacencyMatrix(self.rdmol)

    def get_fragments(self):
        """ Return mol block for fragments """
        fragments = []
        for frag in Chem.GetMolFrags(self.rdmol, asMols=True, sanitizeFrags=False):
            fragments.append(PathMolecule(Chem.MolToMolBlock(frag, confId=-1)))
        return fragments
    
    def embed_molecule(self, seed: int = 31, method='simple'):
        """ """
        fragments = self.get_fragments()
        if len(self.get_fragments()) == 1:
            return PathMolecule(Chem.MolToMolBlock(embed.embed_fragment(self.rdmol)))
        
        if len(self.get_fragments()) == 2:
            embeded_rdfragments = []
            for frag in fragments:
                frag_rdmol = embed.embed_fragment(frag.rdmol)
                embeded_rdfragments(frag_rdmol)
            
            # TODO: EMBED
        else:
            raise NotImplementedError("Can we embed more than 2 fragments?")
            

class ReactionPath:

    __slots__ = ["_reactant_mol", "_product_mol"]

    def __init__(self, reactant: PathMolecule, product: PathMolecule):
        
        self._reactant_mol = reactant
        self._product_mol = product

    def broken_bonds_idx(self):
        """ Return atom pair idx for which the bond inbetween is broken/formed. """
        bond_change_matrix = self._reactant_mol.ac_matrix - self._product_mol.ac_matrix
        bond_change_idx = np.where(np.triu(bond_change_matrix) != 0)
        return list(zip(*bond_change_idx))

    def potential_barrierless(self):
        """ Is it a pure dissociation/association reaction. """
        bond_change_matrix = self._reactant_mol.ac_matrix - self._product_mol.ac_matrix
        bond_change_idx = np.where(np.triu(bond_change_matrix) != 0)
        bond_changes = bond_change_matrix[bond_change_idx]
        return np.all(bond_changes == bond_changes[0])
    
