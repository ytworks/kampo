from rdkit import Chem
from tqdm import tqdm


def read_pdb_frames(pdb_file):
    frames = []
    current_frame = []
    with open(pdb_file, 'r') as file:
        for line in tqdm(file):
            if line.startswith("MODEL"):
                current_frame = []
            elif line.startswith("ENDMDL"):
                frames.append(''.join(current_frame))
            else:
                current_frame.append(line)
    return [Chem.MolFromPDBBlock(frame) for frame in frames]
