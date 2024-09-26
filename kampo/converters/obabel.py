import subprocess

def pdbqt2pdb(pdbqt, pdb):
    qt = f"obabel -ipdbqt {pdbqt} -opdb -O{pdb}"
    subprocess.run(qt, shell=True)
    
def pdbqt2sdf(pdbqt, sdf):
    qt = f"obabel -ipdbqt {pdbqt} -osdf -O{sdf}"
    subprocess.run(qt, shell=True)