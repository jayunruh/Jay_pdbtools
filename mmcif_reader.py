import Bio.PDB
import pandas as pd
#import numpy as np

def getMMCIFpdbdf(fname):
    '''
    this function uses biopython to read an mmcif file into a pdb dataframe
    '''
    parser=Bio.PDB.MMCIFParser()
    struct=parser.get_structure('mystruct',fname)
    pdbdfs=[]
    for model in struct.get_models():
        pdbdf=pd.DataFrame(columns=['type','atom','atype','resname','chain','residue',
            'x','y','z','unk','temp','element'])
        for chain in model.get_chains():
            for residue in chain.get_residues():
                ishet=residue.id[0].startswith('H_')
                atype='ATOM'
                if(ishet):
                    atype='HETATM'
                for atom in residue.get_atoms():
                    temp=[atype,atom.get_serial_number(),atom.name,residue.resname,chain.id,
                          residue.id[1],atom.coord[0],atom.coord[1],atom.coord[2],1,atom.bfactor,atom.element]
                    pdbdf.loc[len(pdbdf)]=temp
        pdbdfs.append(pdbdf)
    return pdbdfs

def getpdbdf(fname):
    '''
    this function uses biopython to read a pdb file into a pdb dataframe
    '''
    parser=Bio.PDB.PDBParser()
    struct=parser.get_structure('mystruct',fname)
    pdbdfs=[]
    for model in struct.get_models():
        pdbdf=pd.DataFrame(columns=['type','atom','atype','resname','chain','residue',
            'x','y','z','unk','temp','element'])
        for chain in model.get_chains():
            for residue in chain.get_residues():
                ishet=residue.id[0].startswith('H_')
                atype='ATOM'
                if(ishet):
                    atype='HETATM'
                for atom in residue.get_atoms():
                    temp=[atype,atom.get_serial_number(),atom.name,residue.resname,chain.id,
                          residue.id[1],atom.coord[0],atom.coord[1],atom.coord[2],1,atom.bfactor,atom.element]
                    pdbdf.loc[len(pdbdf)]=temp
        pdbdfs.append(pdbdf)
    return pdbdfs
