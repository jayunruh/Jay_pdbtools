import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import scipy.spatial as ss
import scipy.ndimage as ndi
import numpy as np
import py3Dmol
import re
from io import StringIO
import subprocess
import jpdbtools2 as jpt2

def findBadLine(lines):
    '''
    this finds the first line that can't be space delimited and returns that position
    '''
    return jpt2.findBadLine(lines)

def fixPDB(fname):
    '''
    this converts a pdb file into a space delimitable string (can't be read elsewhere)
    '''
    return jpt2.fixPDB(fname)

def getpdbdf(fpath,skiprows=0):
    '''
    this is for pdb files with one model and no header
    for models with a header need to search for the initial row (don't do that for now)
    '''
    return jpt2.getpdbdf(fpath,skiprows)

def enforcedtypes(pdbdf):
    '''
    #this automatically forces atom, residue, x,y,z,temp into numeric data types
    '''
    return jpt2.enforcedtypes(pdbdf)

def cleanpdbdf(pdbdf):
    '''
    removes non-atom lines and enforces data types
    '''
    return jpt2.cleanpdbdf(pdbdf)

def writepdbdf(df,outpath=None,verbose=False):
    '''
    #this needs to be a space delimited aligned file whereonly x,y,z,unk, and temp are floating point
    #the coordinates have 3 decimals and unk and temp have 2
    #the atype is right justified unless it has >3 chars (necessary?)
    #there are two spaces to the far right
    '''
    return jpt2.writepdbdf(df,outpath,verbose)

def getFASTA(fpath):
    '''
    reads a fasta file and returns a data frame and a list of lines
    '''
    return jpt2.getFASTA(fpath)

def getFASTA2(lines):
    '''
    reads a FASTA file from it's lines and returns a dataframe
    '''
    return jpt2.getFASTA2(lines)

def writeA3M(fpath,wtid,wtseq,ids,seqs):
    '''
    #this writes an a3m file which is just a fasta with no line returns in the sequence
    '''
    jpt2.writeA3M(fpath,wtid,wtseq,ids,seqs)
    return

def splitDF(df,splitcol,splitval):
    '''
    this splits a dataframe vertically like you would split a string horizontally
    splits happend at splitval instances in splitcol
    returns a list of dataframes
    '''
    return jpt2.splitDF(df,splitcol,splitval)

def processDocking(pdbdf,keywords):
    '''
    #need to split up all of the ligands and calculate their centers of mass
    #try to pull out energy and deltaG for each one
    #the end of each ligand will be the TER type
    returns the target dataframe, ligand dataframes, centers of mass, and key values
    '''
    return jpt2.processDocking(pdbdf,keywords)

#get just the backbones
def getBackbone(pdbdf):
    return jpt2.getBackbone(pdbdf)

def getCA(pdbdf):
    return jpt2.getCA(pdbdf)

def plotClusters(pdbdf,cluster1,cluster2,name1,name2,color1='red',color2='blue'):
    '''
    plots clusters of docking coordinate coms on a pdbdf
    returns the plotly fig
    '''
    resnames=pdbdf['resname']+pdbdf['residue'].astype(str)
    fig=go.Figure()
    wtdata=go.Scatter3d(x=pdbdf['x'],y=pdbdf['y'],z=pdbdf['z'],
        line=dict(color='gray',width=5),mode='lines',
        hovertext=resnames)
    c1data=go.Scatter3d(name=name1,x=cluster1[:,0],y=cluster1[:,1],z=cluster1[:,2],
           marker=dict(size=3,opacity=0.25,color=color1),mode='markers')
    c2data=go.Scatter3d(name=name2,x=cluster2[:,0],y=cluster2[:,1],z=cluster2[:,2],
           marker=dict(size=3,opacity=0.25,color=color2),mode='markers')
    fig.add_traces([wtdata,c1data,c2data])
    fig.show()
    return fig

def plotCompare(pdbdf,mutdf,name1,name2,color1='red',color2='blue'):
    '''
    visually compare two pdbdfs (pdbdf and mutdf)
    returns the plotly fig
    '''
    resnames=pdbdf['resname']+pdbdf['residue'].astype(str)
    mresnames=mutdf['resname']+mutdf['residue'].astype(str)
    fig=go.Figure()
    wtdata=go.Scatter3d(x=pdbdf['x'],y=pdbdf['y'],z=pdbdf['z'],
        line=dict(color=color1,width=5),mode='lines',
        hovertext=resnames,name=name1)
    mutdata=go.Scatter3d(x=mutdf['x'],y=mutdf['y'],z=mutdf['z'],
        line=dict(color=color2,width=5),mode='lines',
        hovertext=mresnames,name=name2)
    fig.add_traces([wtdata,mutdata])
    fig.show()
    return fig

def plotCompare3Dmol(pdbdfs):
    '''
    visually compare two pdbdfs in a list
    returns a plot3D view
    '''
    #this is a plot comparison in cartoon mode in py3Dmol
    view = py3Dmol.view(width=600, height=600)
    colors=['white','blue','green','red','magenta','yellow','cyan']
    for i in range(len(pdbdfs)):
        view.addModel(writepdbdf(pdbdfs[i]),'pdb')
    for i in range(len(pdbdfs)):
        view.setStyle({'model':i},{'cartoon':{'color':colors[i%7]}})
    view.zoomTo()
    return view

def getSeq(cadf):
    '''
    #this get's the AA sequence from a c alpha pdb dataframe
    returns the sequence string and list of residues
    '''
    return jpt2.getSeq(cadf)

def getPaddedStr(intval,padlen):
    return jpt2.getPaddedStr(intval,padlen)

def writeNdx(fname,label,poss,ncols=15):
    jpt2.writeNdx(fname,label,poss,ncols)
    return

def getRMSD(pdbdf1,pdbdf2):
    #assume that the atom sets are identical here (e.g. a set of ca values would be easiest)
    return jpt2.getRMSD(pdbdf1,pdbdf2)

def alignFullRMSD(pdbdf1,pdbdf2,aset='ca'):
    '''
    #this aligns two full atom sets together referencing 'ca' or 'backbone' or 'all' subsets
    #if reslist is not "None" that defines ths subset of atoms that are aligned
    '''
    return jpt2.alignFullRMSD(pdbdf1,pdbdf2,aset)

def transformpdbdf(pdbdf,com,trans=None,comdest=None):
    '''
    #this takes a center of mass and transformation from a subset
    #subtract the center of mass from the full data frame
    #then run the transformation
    '''
    return jpt2.transformpdbdf(pdbdf,com,trans,comdest)

def alignRMSD(pdbdf1,pdbdf2):
    '''
    #this aligns two identical sets of atoms to one another with best fit and returns the RMSD
    #assume that the atom sets are identical here
    #return aligned_df1,aligned_df2,transformation,rmsd,center1,center2
    '''
    return jpt2.alignRMSD(pdbdf1,pdbdf2)

def alignShiftRMSD(pdbdf1,pdbdf2,minshift=-10,maxshift=10):
    '''
    attempts to align two pdbdfs of different length with offset shifts
    brute force search approach minimizing RMSD
    returns minrmsd, bestshift, aligned pdbdf1, and aligned pdbdf2
    '''
    return jpt2.alignShiftRMSD(pdbdf1,pdbdf2,minshift,maxshift)

def getMolecularSurface(pdbdf,resolution=0.25,vdwrad=2.6):
    '''
    Takes a pdbdf and turns it into a 3d mask on a grid determined by resolution (angstroms) and protein extent
    returns the surface (3d ndarray) and z,y,x start coordinates
    it's important that the pdbdf coming in is "clean" in that it has float x,y,z coordinates
    we make the 3D boolean surface using the scipy euclidian distance transform
    the surface is a 3D boolean mask with spatial resolution given by that parameter
    the surface starts at the minimum coordinate minus vdwrad and goes to the max coordinate plus vdwrad
    set the van der waals radius to around 1.2 for accessible surface, and 2.6 for solvent accessible (assumes water radius is 1.4)
    '''
    return jpt2.getMolecularSurface(pdbdf,resolution,vdwrad)

def getSurfaceDistance(coords,surf,spanstart,resolution=0.25):
    '''
    gets the distance to a pre-calculated molecular surface which is a true/false array
    coords is a numpy array of z,y,x float arrays
    spanend=spanstart+surf.shape*resolution
    '''
    return jpt2.getSurfaceDistance(coords,surf,spanstart,resolution)

def getAllDihedrals(pdbdf):
    #assume all one chain and no missing residues
    return jpt2.getAllDihedrals(pdbdf)

def getDihedrals(pdbdf,resnum):
    #dihedrals are phi (prev_C, N, Ca, C)
    #psi: (N,Ca,C,next_N)
    #omega: (Ca,C,next_N,Next_Ca)
    return jpt2.getDihedrals(pdbdf,resnum)

def getDihedral(coords):
    #this gets the dihedral angle for a set of 4 coordinates
    #first get the normals for the two sets of 3 coordinates
    #the normals are given by the cross product of the neighboring vectors
    return jpt2.getDihedral(coords)

def getSecondaryStructures(dihedrals):
    #here we estimate the secondary structures from the dihedrals (fairly poor accuracy)
    return jpt2.getSecondaryStructures(dihedrals)

def getStrideSS(pdbpath):
    '''
    uses the binary program stride shipped with vmd to get secondary structures from a pdb file
    returns the secondary structure list and the sequence
    '''
    return jpt2.getStrideSS(pdbpath)

def calculate_lDDT(pdbdf1,pdbdf2):
    '''
    this calculates the lDDT (local distance difference test) score for two (cleaned) pdb structures
    this metric is translation and rotation invariant (Mariani et al., Bioinformatics, 2013)
    pdbdf1 and pdbdf2 are the two dataframes of the two structures
    pdbdf1 and pdbdf2 should have the same number of rows (atoms)
    returns a dataframe with the lDDT score for each atom and the total number of measured distances (over 4 tolerances)
    '''
    return jpt2.calculate_lDDT(pdbdf1,pdbdf2)
