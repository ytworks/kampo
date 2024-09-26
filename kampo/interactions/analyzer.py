from plip.exchange.report import BindingSiteReport
from plip.structure.preparation import PDBComplex, LigandFinder
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import openbabel
import pandas as pd
from pathlib import Path
from rdkit import Chem
import time
import warnings
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile

# import nglview as nv
# import MDAnalysis as mda
# from MDAnalysis.analysis import rms, diffusionmap, align
# from MDAnalysis.analysis.distances import dist
# from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA

warnings.filterwarnings("ignore")

'''
def run_hydrogenbond_analysis_ppi(md_universe, chain1='A', chain2='B'):
    acceptors_sel = ''
    hydrogens_sel = ''
    donors_sel = ''
    # protein_atoms = md_universe.select_atoms(f"protein")
    protein_b_atoms = md_universe.select_atoms(
        f"(around 3 chainID {chain1}) and chainID {chain2}")
    protein_a_atoms = md_universe.select_atoms(
        f"(around 3 chainID {chain2}) and chainID {chain1}")
    for protein_atoms in [protein_b_atoms, protein_a_atoms]:
        for p in protein_atoms:
            if p.type in ['H']:
                if hydrogens_sel != '':
                    hydrogens_sel += ' or '
                ope = f"(resid {p.resid} and name {p.name})"
                hydrogens_sel += ope
    for protein_atoms in [protein_b_atoms, protein_a_atoms]:
        for p in protein_atoms:
            if p.type in ['H']:
                if donors_sel != '':
                    donors_sel += ' or '
                ope = f"(resid {p.resid} and name {p.name})"
                donors_sel += ope
    for protein_atoms in [protein_b_atoms, protein_a_atoms]:
        for p in protein_atoms:
            if p.type in ['F', 'N', 'O', 'S']:
                if acceptors_sel != '':
                    acceptors_sel += ' or '
                ope = f"(resid {p.resid} and name {p.name})"
                acceptors_sel += ope
    hbonds = HBA(
        universe=md_universe,
        donors_sel=donors_sel,
        hydrogens_sel=hydrogens_sel,
        acceptors_sel=acceptors_sel,
        d_h_cutoff=1.2,
        d_a_cutoff=3.0,
        d_h_a_angle_cutoff=120,
    )
    return hbonds


def run_hydrogenbond_analysis_around_ligand(md_universe, ligand_name):
    """_summary_

    Args:
        md_universe (universe class): MD Universe
        ligand_name (str): Ligand Name

    Returns:
        _type_: bond class
    """
    # リガンド原子 = acceptor sel
    ligand_atoms = md_universe.select_atoms(f"(resname {ligand_name})")
    protein_a_atoms = md_universe.select_atoms(
        f"(around 3 resname {ligand_name}) and protein")
    acceptors_sel = ''
    hydrogens_sel = ''
    donors_sel = ''
    for protein_atoms in [protein_a_atoms, ligand_atoms]:
        for p in protein_atoms:
            if p.type in ['H']:
                if hydrogens_sel != '':
                    hydrogens_sel += ' or '
                ope = f"(resid {p.resid} and name {p.name})"
                hydrogens_sel += ope
    for protein_atoms in [protein_a_atoms, ligand_atoms]:
        for p in protein_atoms:
            if p.type in ['H']:
                if donors_sel != '':
                    donors_sel += ' or '
                ope = f"(resid {p.resid} and name {p.name})"
                donors_sel += ope
    for protein_atoms in [protein_a_atoms, ligand_atoms]:
        for p in protein_atoms:
            if p.type in ['F', 'N', 'O', 'S']:
                if acceptors_sel != '':
                    acceptors_sel += ' or '
                ope = f"(resid {p.resid} and name {p.name})"
                acceptors_sel += ope

    hbonds = HBA(universe=md_universe,
                 donors_sel=donors_sel,
                 hydrogens_sel=hydrogens_sel,
                 acceptors_sel=acceptors_sel,
                 d_h_cutoff=1.2,
                 d_a_cutoff=3.0,
                 d_h_a_angle_cutoff=120,
                 )
    hbonds.run()
    return hbonds


def run_hydrogenbond_analysis(
    universe,
    donor_resid,
    donor_atom,
    hydrogen_atom,
    acceptor_resid,
    acceptor_atom,
    h_cutoff=1.2,
    a_cutoff=3.0,
    angle_cutoff=120,
):
    """
    Perform an hydrogen bond analysis between selcted atom groups.
    Return the data as readable, formatted data frame.

    Parameters
    ----------
    universe: MDAnalysis.core.universe.Universe
        MDAnalysis universe.
    donor_resid: str
        Residue Id of residue containing donor atom in MDAnalysis universe.
    donor_atom: str
        Name of the donor atom in the MDAnalysis universe.
    hydrogen_atom: str
        Name of involved hydrogen atom in the MDAnalysis universe.
    acceptor_resid: str
        Residue Id of residue containing acceptor atom in the MDAnalysis universe.
    acceptor_atom: str
        Name of the acceptor atom in the MDAnalysis universe.
    h_cutoff: float, optional
        Distance cutoff used for finding donor-hydrogen pairs.
    a_cutoff: float, optional
        Donor-acceptor distance cutoff for hydrogen bonds.
    angle_cutoff: int, optional
        D-H-A angle cutoff for hydrogen bonds.

    Returns
    -------
    bond : pandas.core.frame.DataFrame
        DataFrame containing hydrogen bond information over time.
    """

    # set up and run the hydrogen bond analysis (HBA)
    hbonds = HBA(
        universe=universe,
        donors_sel="resid " + donor_resid + " and name " + donor_atom,
        hydrogens_sel="resid " + donor_resid + " and name " + hydrogen_atom,
        acceptors_sel="resid " + acceptor_resid + " and name " + acceptor_atom,
        d_h_cutoff=h_cutoff,
        d_a_cutoff=a_cutoff,
        d_h_a_angle_cutoff=angle_cutoff,
    )
    hbonds.run()
    # extract bond data
    bond = pd.DataFrame(np.round(hbonds.hbonds, 2))
    # name columns (according to MDAnalysis function description)
    bond.columns = [
        "frame",
        "donor_index",
        "hydrogen_index",
        "acceptor_index",
        "distance",
        "angle",
    ]
    # set frame as the index
    bond["frame"] = bond["frame"].astype(int)
    bond.set_index("frame", inplace=True, drop=True)
    return bond


def RMSD_dist_frames(universe, selection):
    """Calculate the RMSD between all frames in a matrix.

    Parameters
    ----------
    universe: MDAnalysis.core.universe.Universe
        MDAnalysis universe.
    selection: str
        Selection string for the atomgroup to be investigated, also used during alignment.

    Returns
    -------
    array: np.ndarray
        Numpy array of RMSD values.
    """
    pairwise_rmsd = diffusionmap.DistanceMatrix(universe, select=selection)
    pairwise_rmsd.run()
    return pairwise_rmsd.dist_matrix


def rmsd_for_atomgroups(universe, selection1, selection2=None):
    """Calulate the RMSD for selected atom groups.

    Parameters
    ----------
    universe: MDAnalysis.core.universe.Universe
        MDAnalysis universe.
    selection1: str
        Selection string for main atom group, also used during alignment.
    selection2: list of str, optional
        Selection strings for additional atom groups.

    Returns
    -------
    rmsd_df: pandas.core.frame.DataFrame
        DataFrame containing RMSD of the selected atom groups over time.
    """

    universe.trajectory[0]
    ref = universe
    rmsd_analysis = rms.RMSD(
        universe, ref, select=selection1, groupselections=selection2)
    rmsd_analysis.run()
    columns = [selection1, *selection2] if selection2 else [selection1]
    rmsd_df = pd.DataFrame(
        np.round(rmsd_analysis.rmsd[:, 2:], 2), columns=columns)
    rmsd_df.index.name = "frame"
    return rmsd_df

'''


def retrieve_plip_interactions_from_mol(pdb_mol):
    temp_file = tempfile.NamedTemporaryFile(
        mode='w+', delete=False, suffix='.pdb')
    Chem.MolToPDBFile(pdb_mol, temp_file.name)
    return retrieve_plip_interactions(temp_file.name)


'''
reference: https://projects.volkamerlab.org/teachopencadd/all_talktorials.html
'''


def retrieve_plip_interactions_for_ppi(pdb_file, chains):
    """
    Retrieves the interactions from PLIP.

    Parameters
    ----------
    pdb_file :
        The PDB file of the complex.

    chains :
        The chains of the complex.

    Returns
    -------
    dict :
        A dictionary of the binding sites and the interactions.
    """
    protlig = PDBComplex()
    protlig.load_pdb(pdb_file)
    lf = LigandFinder(protlig.protcomplex, protlig.altconf,
                      protlig.modres, protlig.covalent, protlig.Mapper)
    ppiligs = []
    for chain in chains:
        ppiligs.append(lf.getpeptides(chain))
    for ligand in ppiligs:
        protlig.characterize_complex(ligand)

    sites = {}
    # loop over binding sites
    for key, site in sorted(protlig.interaction_sets.items()):
        # collect data about interactions
        binding_site = BindingSiteReport(site)
        # tuples of *_features and *_info will be converted to pandas DataFrame
        keys = (
            "hydrophobic",
            "hbond",
            "waterbridge",
            "saltbridge",
            "pistacking",
            "pication",
            "halogen",
            "metal",
        )
        # interactions is a dictionary which contains relevant information for each
        # of the possible interactions: hydrophobic, hbond, etc. in the considered
        # binding site. Each interaction contains a list with
        # 1. the features of that interaction, e.g. for hydrophobic:
        # ('RESNR', 'RESTYPE', ..., 'LIGCOO', 'PROTCOO')
        # 2. information for each of these features, e.g. for hydrophobic
        # (residue nb, residue type,..., ligand atom 3D coord., protein atom 3D coord.)
        interactions = {
            k: [getattr(binding_site, k + "_features")] +
            getattr(binding_site, k + "_info")
            for k in keys
        }
        sites[key] = interactions
    return sites


def retrieve_plip_interactions(pdb_file):
    """
    Retrieves the interactions from PLIP.

    Parameters
    ----------
    pdb_file :
        The PDB file of the complex.

    Returns
    -------
    dict :
        A dictionary of the binding sites and the interactions.
    """
    protlig = PDBComplex()
    protlig.load_pdb(pdb_file)  # load the pdb file
    for ligand in protlig.ligands:
        # find ligands and analyze interactions
        protlig.characterize_complex(ligand)
    sites = {}
    # loop over binding sites
    for key, site in sorted(protlig.interaction_sets.items()):
        # collect data about interactions
        binding_site = BindingSiteReport(site)
        # tuples of *_features and *_info will be converted to pandas DataFrame
        keys = (
            "hydrophobic",
            "hbond",
            "waterbridge",
            "saltbridge",
            "pistacking",
            "pication",
            "halogen",
            "metal",
        )
        # interactions is a dictionary which contains relevant information for each
        # of the possible interactions: hydrophobic, hbond, etc. in the considered
        # binding site. Each interaction contains a list with
        # 1. the features of that interaction, e.g. for hydrophobic:
        # ('RESNR', 'RESTYPE', ..., 'LIGCOO', 'PROTCOO')
        # 2. information for each of these features, e.g. for hydrophobic
        # (residue nb, residue type,..., ligand atom 3D coord., protein atom 3D coord.)
        interactions = {
            k: [getattr(binding_site, k + "_features")] +
            getattr(binding_site, k + "_info")
            for k in keys
        }
        sites[key] = interactions
    return sites


def create_df_from_binding_site_for_ppi(selected_site_interactions, src_chain, dst_chain,
                                        interaction_type="hbond",
                                        ):
    df = create_df_from_binding_site(
        selected_site_interactions, interaction_type=interaction_type)
    df = df[(df['RESCHAIN'] == src_chain) & (df['RESCHAIN_LIG'] == dst_chain)]
    return df


def create_df_from_binding_site(selected_site_interactions, interaction_type="hbond"):
    """
    Creates a data frame from a binding site and interaction type.

    Parameters
    ----------
    selected_site_interactions : dict
        Precaluclated interactions from PLIP for the selected site
    interaction_type : str
        The interaction type of interest (default set to hydrogen bond).

    Returns
    -------
    pd.DataFrame :
        DataFrame with information retrieved from PLIP.
    """

    # check if interaction type is valid:
    valid_types = [
        "hydrophobic",
        "hbond",
        "waterbridge",
        "saltbridge",
        "pistacking",
        "pication",
        "halogen",
        "metal",
    ]

    if interaction_type not in valid_types:
        print("!!! Wrong interaction type specified. Hbond is chosen by default!!!\n")
        interaction_type = "hbond"

    df = pd.DataFrame.from_records(
        # data is stored AFTER the column names
        selected_site_interactions[interaction_type][1:],
        # column names are always the first element
        columns=selected_site_interactions[interaction_type][0],
    )
    return df
