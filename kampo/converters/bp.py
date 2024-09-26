import sys
from rdkit import Chem
from Bio import PDB

def getComplex(ligand_file,
               protein_file,
               output_file):

    # SDF ファイルから小分子を読み込み
    suppl = Chem.SDMolSupplier(ligand_file, removeHs=False)
    ligand = suppl[0]
    if ligand is None:
        print("Error: Couldn't read ligand.")
        sys.exit(1)

    # PDB ファイルからタンパク質を読み込み
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', protein_file)

    # 小分子を PDB 形式に変換
    ligand_pdb = Chem.MolToPDBBlock(ligand)

    # タンパク質と小分子を一緒に書き出し
    with open(output_file, 'w') as f:
        io = PDB.PDBIO()
        io.set_structure(structure)
        io.save(f, write_end=False)  # END レコードを書き出さない
        f.write('TER\n')  # タンパク質と小分子の間にTERレコードを挿入
        f.write(ligand_pdb)

    print("Done! Complex saved to", output_file)