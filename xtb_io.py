import pkgutil
import string

import numpy as np

def read_energy(out) -> dict[str, float]:
    """Read energies from xTB output"""
    energy_order = {0: "elec_energy", 1: "enthalpy", 2: "free_energy"}
    for energy_block in out.strip().split("-----------"):
        if "TOTAL ENERGY" in energy_block:
            break

    energy_idx = 0
    energies = {}
    for line in energy_block.split("\n")[1:]:
        if "GRADIENT NORM" in line:
            break
        energies[energy_order[energy_idx]] = float(line.split()[-3]) * 627.503
        energy_idx += 1
    return energies


def read_opt_structure(out) -> np.ndarray:
    """Read optimized strucure from xTB output"""

    final_structure = out.split("final structure")[-1]
    final_structure = final_structure.split("\n")
    del final_structure[:3]

    coordinates = []
    for line in final_structure:
        if "$end" in line:
            break
        coordinates.append(list(map(float, line.split()[:3])))
    return np.asarray(coordinates, dtype=np.float32)


def read_xtb_path(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Read coordinates and energies from the path search"""

    with open(filename, "r") as path_file:
        xtbpath = path_file.read()

    path_xyz_blocks = xtbpath.split("SCF done")
    natoms = int(path_xyz_blocks[0])
    del path_xyz_blocks[0]

    path_coords = np.zeros((len(path_xyz_blocks), natoms, 3))
    relative_energies = np.zeros(len(path_xyz_blocks))
    for structure_idx, path_strucrure in enumerate(path_xyz_blocks):
        xyz_data = path_strucrure.split("\n")
        relative_energies[structure_idx] = float(xyz_data[0])
        del xyz_data[0]

        coords = np.zeros((natoms, 3))
        for j in range(natoms):
            atom_coord = [coord for coord in xyz_data[j].split()][1:]
            coords[j] = np.array(atom_coord).astype(float)
        path_coords[structure_idx] = coords

    return relative_energies, path_coords


def write_xyz(coords, atomic_symbols) -> str:
    """Write xyz file. If filname write to file"""

    xyz = f"{len(atomic_symbols)}\n\n"
    for symbol, coord in zip(atomic_symbols, coords):
        coord = [float(x) for x in coord]
        xyz += f"{symbol}  {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}\n"
    return xyz


def get_rmsd_template():
    """ Writes the input file text needed for the xtb path search """
        
    rmsd_template = string.Template(
        pkgutil.get_data(__name__, "templates/rmsdpp_template").decode('utf-8')
    )
    return rmsd_template


def write_scan_input(bond_identifier, bond_distances):
    """ """
    scaninp_txt = "$scan\n"
    scaninp_txt += "    mode=concerted\n"
    for (atom_i, atom_j), bond_dist in zip(bond_identifier, bond_distances):
        scaninp_txt += f"    distance: {atom_i + 1},{atom_j + 1},{bond_dist}; "
        scaninp_txt += f"{bond_dist},{bond_dist + 2},40\n"
    scaninp_txt += "$end"

    return scaninp_txt
