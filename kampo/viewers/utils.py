from ..interactions import analyzer
import pandas as pd
import py3Dmol
import tempfile
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import AllChem
from rdkit import RDConfig
import os


pharmacophore_colors = {
    "Donor": (0, 255, 0),        # 緑色
    "Acceptor": (255, 0, 0),     # 赤色
    "NegIonizable": (0, 0, 255),  # 青色
    "PosIonizable": (255, 255, 0),  # 黄色
    "Aromatic": (128, 0, 128),   # 紫色
    "ZnBinder": (0, 255, 255),   # シアン色
    "Hydrophobe": (255, 165, 0),  # オレンジ色
    "LumpedHydrophobe": (255, 105, 180)  # ピンク色
}


def show_3d_interactions_from_mol(pdb_mol, ligand_name='UNL'):
    temp_file = tempfile.NamedTemporaryFile(
        mode='w+', delete=False, suffix='.pdb')
    Chem.MolToPDBFile(pdb_mol, temp_file.name)
    return show_3d_interactions(temp_file.name, ligand_name)


def view_pharmacophore_from_mol(pdb_mol,
                                smiles='O=C(O)CCCN1CC(Oc2c1cccc2NC(=O)c1ccc(cc1)OCCCCCc1ccccc1)C(=O)O',
                                ligand_name="LIG"):
    view, complete_dfs = show_3d_interactions_from_mol(pdb_mol, ligand_name)
    temp_file = tempfile.NamedTemporaryFile(
        mode='w+', delete=False, suffix='.pdb')
    Chem.MolToPDBFile(pdb_mol, temp_file.name)
    ligand_mol = extract_ligand_from_pdb(temp_file.name,
                                         smiles=smiles,
                                         residue_name=ligand_name)
    pcos = extract_pharmacophore(ligand_mol=ligand_mol)
    view = add_sphere(view, pcos)
    df = pd.DataFrame(
        pcos, columns=['Type', 'Detail', 'AtomNumber', 'X', 'Y', 'Z'])

    return view, complete_dfs, df


def add_sphere(view, pcos):
    for pco in pcos:
        color = pharmacophore_colors[pco[0]]
        view.addSphere({'center': {'x': pco[3], 'y': pco[4], 'z': pco[5]},
                        'radius': 0.5,
                        'color': f'rgb({color[0]},{color[1]},{color[2]})',
                        "hoverable": True,
                        "hover_callback": '''function(atom,viewer,event,container) {
                                                if(!this.label) {
                                                    this.label = viewer.addLabel("%s = (%s, %s, %s)",{position: this, backgroundColor: 'mintcream', fontColor:'black'});
                                            }}''' % (pco[0], pco[3], pco[4], pco[5]),
                        "unhover_callback": '''function(atom,viewer) { 
                                    if(this.label) {
                                        viewer.removeLabel(this.label);
                                        delete this.label;
                                    }
                                    }'''})
    return view


def view_pharmacophore(pdb_file,
                       smiles='O=C(O)CCCN1CC(Oc2c1cccc2NC(=O)c1ccc(cc1)OCCCCCc1ccccc1)C(=O)O',
                       ligand_name="LIG"):
    view, complete_dfs = show_3d_interactions(pdb_file, ligand_name)
    ligand_mol = extract_ligand_from_pdb(pdb_file,
                                         smiles=smiles,
                                         residue_name=ligand_name)
    pcos = extract_pharmacophore(ligand_mol=ligand_mol)
    view = add_sphere(view, pcos)
    df = pd.DataFrame(
        pcos, columns=['Type', 'Detail', 'AtomNumber', 'X', 'Y', 'Z'])
    return view, complete_dfs, df


def extract_ligand_from_pdb(pdb_file, smiles='O=C(O)CCCN1CC(Oc2c1cccc2NC(=O)c1ccc(cc1)OCCCCCc1ccccc1)C(=O)O',
                            residue_name="LIG"):
    mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)
    ligand_atoms = [atom.GetIdx() for atom in mol.GetAtoms(
    ) if atom.GetPDBResidueInfo().GetResidueName() == residue_name]

    # リガンド原子のみを含むサブモレキュールを作成
    ligand = Chem.RWMol(mol)
    ligand_atoms_set = set(ligand_atoms)
    for atom_idx in reversed(range(ligand.GetNumAtoms())):
        if atom_idx not in ligand_atoms_set:
            ligand.RemoveAtom(atom_idx)
    lig = ligand.GetMol()
    reference_mol = Chem.MolFromSmiles(smiles)
    ligand_mol = AllChem.AssignBondOrdersFromTemplate(
        reference_mol, lig)
    ligand_mol.AddConformer(lig.GetConformer(0))
    return ligand_mol


def show_3d_interactions_for_ppi(pdb_file, chains, src_chain, dst_chain):
    bond_types = ["hydrophobic",
                  "hbond",
                  "waterbridge",
                  "saltbridge",
                  "pistacking",
                  "pication",
                  "halogen"]
    colors = {"hydrophobic": "red",
              "hbond": "blue",
              "waterbridge": "blueCarbon",
              "saltbridge": "white",
              "pistacking": "yellow",
              "pication": "orange",
              "halogen": "magenta"}
    interactions_by_site = analyzer.retrieve_plip_interactions_for_ppi(
        pdb_file, chains)
    index_of_selected_site = 0
    selected_site = list(interactions_by_site.keys())[index_of_selected_site]
    dfs = []
    complete_dfs = {}
    for bond_type in bond_types:
        df = analyzer.create_df_from_binding_site_for_ppi(
            interactions_by_site[selected_site],
            src_chain=src_chain,
            dst_chain=dst_chain,
            interaction_type=bond_type)
        complete_dfs[bond_type] = df
        df = df[["RESNR", "RESTYPE", "LIGCOO",
                 "PROTCOO", 'RESNR_LIG', "RESTYPE_LIG"]]
        df['BONDTYPE'] = bond_type
        dfs.append(df)
    df = pd.concat(dfs, axis=0)

    view = py3Dmol.view()
    view.addModel(open(pdb_file, 'r').read(), 'pdb')
    view.setStyle({"cartoon": {"color": "grey"}})
    view = hover_atom(view)
    for idx, rows in df.iterrows():
        view.setStyle({'resi': rows["RESNR"]},
                      {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.2}})
        view.setStyle({'resi': rows["RESNR_LIG"]},
                      {'stick': {'colorscheme': 'whiteCarbon', 'radius': 0.2}})
        sx, sy, sz = rows["LIGCOO"]
        ex, ey, ez = rows["PROTCOO"]
        view.addCylinder({"start": dict(x=sx, y=sy, z=sz),
                          "end":   dict(x=ex, y=ey, z=ez),
                          "color": colors[rows["BONDTYPE"]],
                          "radius": .15,
                          "dashed": True,
                          "fromCap": 1,
                          "toCap": 1,
                          "hoverable": True,
                          "hover_callback": '''function(atom,viewer,event,container) {
                                                if(!this.label) {
                                                    this.label = viewer.addLabel("%s",{position: this, backgroundColor: 'mintcream', fontColor:'black'});
                                            }}''' % rows["BONDTYPE"],
                          "unhover_callback": '''function(atom,viewer) { 
                                    if(this.label) {
                                        viewer.removeLabel(this.label);
                                        delete this.label;
                                    }
                                    }'''
                          })
    view.setViewStyle({'style': 'outline', 'color': 'black', 'width': 0.1})

    view.zoomTo()
    return view, complete_dfs


def show_3d_interactions(pdb_file, ligand_name='UNL'):
    """Show 3D interactions of a complex.

    Parameters
    ----------
    pdb_file : str
        Path to a PDB file of a complex.
    """
    bond_types = ["hydrophobic",
                  "hbond",
                  "waterbridge",
                  "saltbridge",
                  "pistacking",
                  "pication",
                  "halogen"]
    colors = {"hydrophobic": "red",
              "hbond": "blue",
              "waterbridge": "blueCarbon",
              "saltbridge": "white",
              "pistacking": "yellow",
              "pication": "orange",
              "halogen": "magenta"}

    interactions_by_site = analyzer.retrieve_plip_interactions(pdb_file)
    index_of_selected_site = 0
    selected_site = list(interactions_by_site.keys())[index_of_selected_site]
    dfs = []
    complete_dfs = {}
    for bond_type in bond_types:
        df = analyzer.create_df_from_binding_site(
            interactions_by_site[selected_site],
            interaction_type=bond_type)
        complete_dfs[bond_type] = df
        df = df[["RESNR", "RESTYPE", "LIGCOO", "PROTCOO"]]
        df['BONDTYPE'] = bond_type
        dfs.append(df)
    df = pd.concat(dfs, axis=0)

    view = py3Dmol.view()
    view.addModel(open(pdb_file, 'r').read(), 'pdb')
    view.setStyle({"cartoon": {"color": "grey"}})
    view = hover_atom(view)
    LIG = [ligand_name]
    view.addStyle({'and': [{'resn': LIG}]},
                  {'stick': {'colorscheme': 'magentaCarbon', 'radius': 0.3}})
    view.setViewStyle({'style': 'outline', 'color': 'black', 'width': 0.1})
    for idx, rows in df.iterrows():
        view.setStyle({'resi': rows["RESNR"]},
                      {'stick': {'colorscheme': 'whiteCarbon', 'radius': 0.2}})
        sx, sy, sz = rows["LIGCOO"]
        ex, ey, ez = rows["PROTCOO"]
        view.addCylinder({"start": dict(x=sx, y=sy, z=sz),
                          "end":   dict(x=ex, y=ey, z=ez),
                          "color": colors[rows["BONDTYPE"]],
                          "radius": .15,
                          "dashed": True,
                          "fromCap": 1,
                          "toCap": 1,
                          "hoverable": True,
                          "hover_callback": '''function(atom,viewer,event,container) {
                                                if(!this.label) {
                                                    this.label = viewer.addLabel("%s",{position: this, backgroundColor: 'mintcream', fontColor:'black'});
                                            }}''' % rows["BONDTYPE"],
                          "unhover_callback": '''function(atom,viewer) { 
                                    if(this.label) {
                                        viewer.removeLabel(this.label);
                                        delete this.label;
                                    }
                                    }'''
                          })
    view.setViewStyle({'style': 'outline', 'color': 'black', 'width': 0.1})

    view.zoomTo()
    return view, complete_dfs


def extract_pharmacophore(ligand_mol):
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

    pcos = []

    feats = factory.GetFeaturesForMol(ligand_mol)
    for i in range(len(feats)):
        pcos.append([feats[i].GetFamily(), feats[i].GetType(), feats[i].GetAtomIds(),
                    feats[i].GetPos()[0], feats[i].GetPos()[1], feats[i].GetPos()[2]])
    return pcos


def hover_atom(view):
    view.setHoverable({}, True, '''function(atom,viewer,event,container) {
                   if(!atom.label) {
                    atom.label = viewer.addLabel(atom.resn+":"+atom.resi,{position: atom, backgroundColor: 'mintcream', fontColor:'black'});
                   }}''',
                      '''function(atom,viewer) { 
                   if(atom.label) {
                    viewer.removeLabel(atom.label);
                    delete atom.label;
                   }
                }''')

    return view
