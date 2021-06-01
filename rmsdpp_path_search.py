import os
import logging
import tempfile
import subprocess
from pathlib import Path
from collections import namedtuple

from subprocess import Popen, PIPE

import numpy as np
from rdkit import Chem

import xtb_io
from xyz2mol_local import xyz2AC_vdW  # Remove this dependency

_logger = logging.getLogger('rmsd_pp')

def run_shell_cmd(cmd, cwd=None):
    """ Function to run xTB program """
    popen = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=True,
        cwd=cwd
    )
    output, err = popen.communicate()
    return output, err 


def get_xtb_version(xtb_path):
    """
    Return the xTB version of xtb_path
    """
    out, _ = run_shell_cmd(f"{xtb_path} --version")
    xtb_version = out.strip().split("\n")[-2].split()[2]
    return xtb_version


class XtbError(Exception):
    """Error related to xTB"""

    pass


class XtbCalculator:
    """Uses xTB to optimize structures and compute energies"""

    tested_xtb_versions = ["6.1.4"]

    def __init__(
        self,
        scr = '_xtb_scr_dir_',
        charge: int = 0,
        spin: int = 1,
        opt: bool = True,
        thermal_energies: bool = False,
        **kwds,
    ):
        """
        NB. ekstra kwds are added to the xTB input. i.e. gbsa=water adds --gbsa water.

        # TODO: make opt an "ekstra" kwd. Such that opt=loose,tight, etc.
        """
        self.scr = Path(scr)
        self._charge = charge
        self._spin = spin
        self._run_opt = opt
        self._thermal_energies = thermal_energies
        self._ekstra_xtb_kwds = kwds  # Solvent, GFN-x,

        # Test that the xTB version is actually valid.
        self._setup_xtb_enviroment()
        xtb_version = get_xtb_version(self._xtb_path)
        if xtb_version not in self.tested_xtb_versions:
            raise XtbError(f"xTB version {xtb_version} is not tested")

        # set scratch dir
        self.scr.mkdir(parents=True, exist_ok=True)

    def _setup_xtb_enviroment(self):
        """
        Setup xTB environment
        """
        os.environ["OMP_NUM_THREADS"] = str(1)
        os.environ["MKL_NUM_THREADS"] = str(1)

        if "XTB_CMD" not in os.environ:
            raise XtbError('XTB_CMD not defined. export XTB_CMD="path to xtb"')

        self._xtb_path = Path(os.environ["XTB_CMD"])
        os.environ["XTBPATH"] = str(self._xtb_path.parents[1])

    def _make_cmd(self, input_filename: str) -> str:
        """ """
        xtb_cmd = f"{self._xtb_path} {input_filename} --norestart"
        xtb_cmd += f" --chrg {self._charge} --uhf {self._spin - 1}"
        xtb_cmd += f" --namespace {input_filename.split('.')[0]}"

        if self._thermal_energies:
            xtb_cmd += " --hess"

        if self._run_opt:
            xtb_cmd += " --opt loose"

        for kwd, key in self._ekstra_xtb_kwds.items():
            xtb_cmd += f" --{kwd} {key}"

        return xtb_cmd

    def _write_input(
        self, atom_symbols: np.ndarray, coords: np.ndarray, namespace: str, path: str
    ) -> str:
        """Write xyz file"""
        if len(atom_symbols) != len(coords):
            raise XtbError("Length of atom_symbols and coords don't match")

        input_filename = namespace + ".xyz"
        with open(path / input_filename, "w") as inputfile:
            inputfile.write(xtb_io.write_xyz(coords, atom_symbols))
        return input_filename

    def __call__(
        self, atom_symbols: np.ndarray, coords: np.ndarray, namespace: str
    ) -> tuple[dict[str, float], np.ndarray]:
        """
        Run xTB calculation for the structure defined by the atom_symbols and coords.
        """
        self._setup_xtb_enviroment()
        with tempfile.TemporaryDirectory(dir=self.scr) as temp:
            inpfile = self._write_input(atom_symbols, coords, namespace, Path(temp))
            xtb_cmd = self._make_cmd(inpfile)
            out, err = run_shell_cmd(xtb_cmd, cwd=temp)

            if err.strip() != "normal termination of xtb":
                raise XtbError("Calculation of {namespace} terminated with error")

        energies = xtb_io.read_energy(out)
        coordinates = xtb_io.read_opt_structure(out)

        return energies, coordinates


class XtbPathSearch:  # TODO rename to XtbPPSearch
    """
    Run xTB path search.
    """

    tested_xtb_versions = ["6.1.4"]

    KPULL_LIST = [-0.02, -0.02, -0.02, -0.03, -0.03, -0.04, -0.04]
    ALP_LIST = [0.6, 0.3, 0.3, 0.6, 0.6, 0.6, 0.4]

    def __init__(self, scr = '_xtb_scr_dir_', charge: int = 0, spin: int = 1, nruns: int = 1, **kwds):
        
        self.scr = Path(scr)
        self._charge = charge
        self._spin = spin

        self._nruns = nruns
        self._ekstra_xtb_kwds = kwds

        self._product_coords = None
        self._reactant_coords = None
        self._atmoic_symbols = []

        # Check the xTB version
        self._setup_xtb_enviroment()
        xtb_version = get_xtb_version(self._xtb_path)
        if xtb_version not in self.tested_xtb_versions:
            raise XtbError(f"xTB version {xtb_version} is not tested")

        # Make scratch dir
        self.scr.mkdir(parents=True, exist_ok=True)

    def _setup_xtb_enviroment(self) -> None:
        """
        Setup xTB environment
        """
        os.environ["OMP_NUM_THREADS"] = str(1)
        os.environ["MKL_NUM_THREADS"] = str(1)

        if "XTB_CMD" not in os.environ:
            raise XtbError('XTB_CMD not defined. export XTB_CMD="path to xtb"')

        self._xtb_path = Path(os.environ["XTB_CMD"])
        os.environ["XTBPATH"] = str(self._xtb_path.parents[1])

    def _write_path_input(self, kpush, kpull, alpha, temp, tempdir) -> None:
        """ Write reactant and product .xyz files, and the path.inp file. """

        # Write .xyz input
        with open(tempdir / "reactant.xyz", 'w') as reactant_xyz:
            reactant_xyz.write(
                xtb_io.write_xyz(self._reactant_coords, self._atmoic_symbols)
            )

        with open(tempdir / "product.xyz", 'w') as product_xyz:
            product_xyz.write(
                xtb_io.write_xyz(self._product_coords, self._atmoic_symbols)
            )

        # Write path input file
        rmsd_template = xtb_io.get_rmsd_template()
        with open(tempdir / "path.inp", 'w') as path_file:
            path_file.write(
                rmsd_template.safe_substitute(
                    kpush=kpush, kpull=kpull, alpha=alpha, temperature=temp
                )
            )

    def _make_xtb_path_cmd(self, forward: bool = True) -> str:
        """Make cmdline cmd that can run the path search"""
        if forward:
            tmp_reac, tmp_prod = "reactant.xyz", "product.xyz"
        else:
            tmp_reac, tmp_prod = "product.xyz", "reactant.xyz"

        cmd = f"{self._xtb_path} {tmp_reac} --path {tmp_prod} --input path.inp --norestart"
        cmd += f" --chrg {self._charge} --uhf {self._spin - 1}"
        _logger.debug(cmd)
        # TODO add ekstra kwds.

        return cmd

    def _read_final_rmsd(self, output):
        """Checks that the path have an RMSD below 0.5 AA."""
        output_lines = output.split("\n")
        for line in reversed(output_lines):
            if "run 1  barrier" in line:
                try:
                    rmsd = float(line.split()[-1])
                except:
                    rmsd = 9999.9  # rmsd above 0.5 reaction not complete.
        return rmsd

    def _is_reac_prod_identical(self, path_coords):
        """
        This function ensures that if RMSD is above 0.5AA, and something
        happend in the last iteration - it is probably an intermediate.
        """
        pt = Chem.GetPeriodicTable()
        atom_nums = [pt.GetAtomicNumber(atom) for atom in self._atmoic_symbols]

        # TODO make your own version of this, so it doesn't rely on xyz2mol.
        reactant_ac, _ = xyz2AC_vdW(atom_nums, path_coords[0])
        product_ac, _ = xyz2AC_vdW(atom_nums, path_coords[-1])

        if np.array_equal(reactant_ac, product_ac):  # Nothing happend - reac = prod
            return True
        return False

    def _run_path(self, kpush, kpull, alpha, temp, forward=True):
        """
        Run a xTB path search for a given parameter set.
        """
        with tempfile.TemporaryDirectory(dir=self.scr) as tempdirname:
            tempdirname = Path(tempdirname)
            self._write_path_input(
                round(kpush, 4), round(kpull, 4), round(alpha, 4), temp, tempdirname
            )
            cmd = self._make_xtb_path_cmd(forward)
            out, err = run_shell_cmd(cmd, cwd=tempdirname)
            try:
                relative_energies, path_coords = xtb_io.read_xtb_path(tempdirname / "xtbpath_1.xyz")
            except:
                pass

        if err.strip() != "normal termination of xtb":
            return None, None, None
        else:
            return out, relative_energies, path_coords

    def _find_xtb_path(self, temperature: float = 300):
        """Run through KPULL and ALP lists and perform a xTB path scan"""

        def run_params_set(kpush, kpull, alpha, forward):
            """A parameter set is run 3 times. Each time it is multiplied by 1.5"""

            PathResults = namedtuple("PathResults", "rmsd energies path")

            run_info = []
            for iter_num in range(3):
                out, path_energies, path_coords = self._run_path(
                    kpush=kpush,
                    kpull=kpull,
                    alpha=alpha,
                    temp=temperature,
                    forward=forward,
                )

                rmsd = self._read_final_rmsd(out)
                run_info.append(PathResults(rmsd, path_energies, path_coords))

                # xTB found a path
                if rmsd <= 0.5:
                    return True, run_info

                kpush *= 1.5
                kpull *= 1.5

            # if nothing happedned
            return False, run_info

        reaction_forward = True
        run_num = 0
        for set_idx, (kpull, alpha) in enumerate(zip(self.KPULL_LIST, self.ALP_LIST)):
            if set_idx == 0:
                kpush = 0.008
            else:
                kpush = 0.01
            found_path, run_info = run_params_set(
                kpush=kpush, kpull=kpull, alpha=alpha, forward=reaction_forward
            )

            if found_path:
                _logger.info(" RMSD-PP found a path")
                return True, run_info[-1]

            if self._is_reac_prod_identical(run_info[-1].path):
                if reaction_forward:
                    self._reactant_coords = run_info[-1].path[-1]
                else:
                    self._product_coords = run_info[-1].path[-1]
            else:
                _logger.error(" RMSD not below 0.5 A. Most likely not a onestep reaction")
                return "found intermediate", None

            run_num += 1
            if run_num % 2 == 0:
                reaction_forward = True
            else:
                reaction_forward = False

        return "increace temp", None

    def _compute_sp_energies(self, path, **kwds):
        """The Path energies doesn't correspond to SP energies.
        To be sure what the energies is, perform SP xTB energies on path.
        """
        xtbcalc = XtbCalculator(
            charge=self._charge, spin=self._spin
        )  # Add ekstra kwds such as solvent
        path_sp_energies = []
        for path_point in path:
            energies, _ = xtbcalc(self._atmoic_symbols, path_point, namespace="sp_calc")
            path_sp_energies.append(energies["elec_energy"])
        return np.asarray(path_sp_energies)

    def run_path_search(
        self,
        reactant_coords: np.ndarray,
        product_coords: np.ndarray,
        atom_symbols: list[str],
    ):
        """
        Run the path search and return pathinfo:
        """
        self._reactant_coords = reactant_coords
        self._product_coords = product_coords
        self._atmoic_symbols = atom_symbols

        return_msg, path_info = self._find_xtb_path(temperature=300)
        if return_msg == "increace temp":
            _logger.info(" Increasing temperature to 6000 K.")
            return_msg, path_info = self._find_xtb_path(temperature=6000)
        elif return_msg == "found intermediate":
            return None

        # If it didn't converge now, stop.
        if return_msg:
            path_info = path_info._asdict()
            path_info["energies"] = self._compute_sp_energies(path_info["path"])
            return path_info


class XtbScanPath:

    tested_xtb_versions = ["6.1.4"]

    def __init__(
        self,
        scr = "_xtb_scr_dir_",
        charge: int = 0,
        spin: int = 1,
        opt: bool = True,
        thermal_energies: bool = False,
        **kwds,
    ):
        """ """

        self._charge = charge
        self._spin = spin
        self._run_opt = opt
        self._thermal_energies = thermal_energies
        self._ekstra_xtb_kwds = kwds  # Solvent, GFN-x,

        # Test that the xTB version is actually valid.
        self._setup_xtb_enviroment()
        xtb_version = get_xtb_version(self._xtb_path)
        if xtb_version not in self.tested_xtb_versions:
            raise XtbError(f"xTB version {xtb_version} is not tested")

    def _setup_xtb_enviroment(self):
        """
        Setup xTB environment
        """
        os.environ["OMP_NUM_THREADS"] = str(1)
        os.environ["MKL_NUM_THREADS"] = str(1)

        if "XTB_CMD" not in os.environ:
            raise XtbError('XTB_CMD not defined. export XTB_CMD="path to xtb"')

        self._xtb_path = Path(os.environ["XTB_CMD"])
        os.environ["XTBPATH"] = str(self._xtb_path.parents[1])

    def _make_cmd(self, input_filename: str) -> str:
        """ """
        xtb_cmd = f"{self._xtb_path} {input_filename} --input scan.inp --norestart"
        xtb_cmd += f" --chrg {self._charge} --uhf {self._spin - 1}"
        xtb_cmd += f" --namespace {input_filename.split('.')[0]}"

        if self._thermal_energies:
            xtb_cmd += " --hess"

        if self._run_opt:
            xtb_cmd += " --opt loose"

        for kwd, key in self._ekstra_xtb_kwds.items():
            xtb_cmd += f" --{kwd} {key}"

        return xtb_cmd

    def _bond_distance(self, coords):
        """ """
        distances = []
        for atom_i_idx, atom_j_idx in self._bonds_to_scan:
            atom_i_pos = coords[atom_i_idx]
            atom_j_pos = coords[atom_j_idx]
            distances.append(round(np.sqrt(np.sum((atom_i_pos - atom_j_pos) ** 2)), 5))
        return np.asarray(distances)

    def _run_reactant_or_product(self):
        """Sum bond distances and"""
        reactant_bond_dist = self._bond_distance(self._reactant_coords)
        product_bond_dist = self._bond_distance(self._product_coords)

        # Check that all distances in one is larger, not just one.
        dist_check = reactant_bond_dist < product_bond_dist
        if not np.all(dist_check == dist_check[0]):
            raise XtbError("Likey not a pure dissociation/association reaction")

        if sum(reactant_bond_dist) < sum(product_bond_dist):
            return "reactant"
        else:
            return "product"

    def _write_path_input(self, tempdir):
        """Write reactant and product .xyz files, and the path.inp file."""

        # Write .xyz input
        with open(tempdir / "reactant.xyz", 'w') as reactant_xyz:
            reactant_xyz.write(
                xtb_io.write_xyz(self._reactant_coords, self._atmoic_symbols)
            )

        with open(tempdir / "product.xyz", 'w') as product_xyz:
            product_xyz.write(
                xtb_io.write_xyz(self._product_coords, self._atmoic_symbols)
            )

        # Write scan input file
        bond_distances = self._bond_distance(self._product_coords)
        with open(tempdir / "scan.inp", 'w') as scan_file:
            scan_file.write(
                xtb_io.write_scan_input(self._bonds_to_scan, bond_distances)
            )

    def run_path_scan(
        self,
        reactant_coords: np.ndarray,
        product_coords: np.ndarray,
        atom_symbols: list[str],
        bonds_to_scan: list[tuple[int, int]],
    ) -> tuple[np.ndarray, np.ndarray]:

        self._reactant_coords = reactant_coords
        self._product_coords = product_coords
        self._atmoic_symbols = atom_symbols
        self._bonds_to_scan = bonds_to_scan

        with tempfile.TemporaryDirectory(dir=".") as tempdirname:
            tempdirname = Path(tempdirname)
            self._write_path_input(tempdirname)
            to_run = self._run_reactant_or_product()
            scan_cmd = self._make_cmd(to_run + ".xyz")

            out, err = run_shell_cmd(scan_cmd, cwd=tempdirname)
            scan_energies, scan_path = xtb_io.read_xtb_path(tempdirname / f"{to_run}.xtbscan.log")

        # If scan is performed from prod -> reac flip path/energies.
        if to_run == "product":
            scan_energies = np.flip(scan_energies)
            scan = np.flip(scan_path, axis=0)

        rel_energy = scan_energies - scan_energies[0]
        print(rel_energy*627.503)
        return scan_energies, scan_path
