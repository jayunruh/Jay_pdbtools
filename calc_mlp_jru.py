#this module is an implementation of the ChimeraX lipophilicity calculator implemented in 3D
#see https://github.com/RBVI/ChimeraX/blob/develop/src/bundles/mlp/src/mlp.py
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time

#here are the residue potential values from chimerax
#combo of initial mlp values and Ghose et al 1998
fidatadefault = {         #Default fi table
 'ALA': {'CB': 0.4395,    #fi : lipophilic atomic potential
         'C': -0.1002,
         'CA': -0.1571,
         'O': -0.0233,
         'N': -0.6149},
 'ARG': {'C': -0.1002,
         'CA': -0.1571,
         'CB':  0.3212,
         'CG':  0.3212,
         'CD':  0.0116,
         'CZ':  0.5142,
         'N': -0.6149,
         'NE': -0.1425,
         'NH1': -0.5995,
         'NH2': -0.5995,
         'O': -0.0233},
 'ASN': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.0348,
         'CG': -0.1002,
         'N': -0.6149,
         'ND2': -0.7185,
         'O': -0.0233,
         'OD1': -0.0233},
 'ASP': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.0348,
         'CG': -0.1002,
         'N': -0.6149,
         'O': -0.0233,
         'OD1': -0.4087,
         'OD2': -0.4087},
 'CYS': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.0116,
         'N': -0.6149,
         'O': -0.0233,
         'SG': 0.5110},
 'GLN': {'C': -0.1002,
         'CA': -0.1571,
         'CB':  0.3212,
         'CG':  0.0348,
         'CD': -0.1002,
         'N': -0.6149,
         'NE2': -0.7185,
         'O': -0.0233,
         'OE1': -0.0233},
 'GLU': {'C': -0.1002,
         'CA': -0.1571,
         'CB':  0.3212,
         'CG':  0.0348,
         'CD': -0.1002,
         'N': -0.6149,
         'O': -0.0233,
         'OE1': -0.4087,
         'OE2': -0.4087},
 'GLY': {'C': -0.1002,
         'CA': -0.2018,
         'O': -0.0233,
         'N': -0.6149},
 'HIS': {'C': -0.1002,
         'CA': -0.1571,
         'CB':  0.0348,
         'CG': 0.2361,
         'CD2': 0.5185,
         'CE1': 0.1443,
         'N': -0.6149,
         'ND1': -0.2660,
         'NE2': -0.2660,
         'O': -0.0233},
 'HYP': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.3212,
         'CG': -0.0504,
         'CD': 0.0116,
         'N': -0.5113,
         'O': -0.0233,
         'OD1': -0.4603},
 'ILE': {'C': -0.1002,
         'CA': -0.1571,
         'CB': -0.0015,
         'CG1': 0.4562,
         'CG2': 0.6420,
         'CD1': 0.6420,
         'N': -0.6149,
         'O': -0.0233},
 'LEU': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.3212,
         'CG': 0.0660,
         'CD1': 0.6420,
         'CD2': 0.6420,
         'N': -0.6149,
         'O': -0.0233},
 'LYS': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.3212,
         'CG': 0.4562,
         'CD': 0.4562,
         'CE': 0.0116,
         'NZ': -0.8535,
         'N': -0.6149,
         'O': -0.0233},
 'MET': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.3212,
         'CG': 0.0116,
         'CE': 0.1023,
         'N': -0.6149,
         'O': -0.0233,
         'SD': 0.5906},
 'MSE': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.3212,
         'CG': 0.0116,
         'CE': 0.1023,
         'N': -0.6149,
         'O': -0.0233,
         'SE': 0.6601},
 'UNK': {'C': -0.1002,
         'CA': -0.1571,
         'N': -0.6149,
         'O': -0.0233},
 'ACE': {'C': -0.1002,
         'CH3': 0.0099,
         'O': -0.0233},
 'NME': {'N': -0.6149,
         'C': 0.1023},
 'NH2': {'N': -0.7185},
 'PCA': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.3212,
         'CG': 0.0348,
         'CD': -0.1002,
         'N': -0.6149,
         'O': -0.0233,
         'OE': -0.0233},
 'PHE': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.3212,
         'CG': 0.1492,
         'CD1': 0.3050,
         'CD2': 0.3050,
         'CE1': 0.3050,
         'CE2': 0.3050,
         'CZ': 0.3050,
         'N': -0.6149,
         'O': -0.0233},
 'PRO': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.3212,
         'CG': 0.3212,
         'CD': 0.0116,
         'N': -0.5113,
         'O': -0.0233},
 'SER': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.0116,
         'N': -0.6149,
         'O': -0.0233,
         'OG': -0.4603},
 'THR': {'C': -0.1002,
         'CA': -0.1571,
         'CB': -0.0514,
         'CG2': 0.4395,
         'N': -0.6149,
         'O': -0.0233,
         'OG1': -0.4603},
 'TRP': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.0348,
         'CG': 0.1492,
         'CD1': 0.5185,
         'CD2': 0.1492,
         'CE2': 0.1539,
         'CE3': 0.3050,
         'CH2': 0.3050,
         'CZ2': 0.3050,
         'CZ3': 0.3050,
         'N': -0.6149,
         'NE1': 0.0223,
         'O': -0.0233},
 'TYR': {'C': -0.1002,
         'CA': -0.1571,
         'CB': 0.3212,
         'CG': 0.1492,
         'CD1': 0.3050,
         'CD2': 0.3050,
         'CE1': 0.3050,
         'CE2': 0.3050,
         'CZ': 0.1539,
         'N': -0.6149,
         'O': -0.0233,
         'OH': -0.1163},
 'VAL': {'C': -0.1002,
         'CA': -0.1571,
         'CB': -0.0015,
         'CG1': 0.6420,
         'CG2': 0.6420,
         'N': -0.6149,
         'O': -0.0233}}

def addfiColumn(pdbdf):
    '''
    this adds the fidata column to a pdbdf based on residue and atom types
    '''
    fidatalist=[]
    for i in range(len(pdbdf)):
        atype=pdbdf.loc[i,'atype']
        if(atype=='OXT'):
            atype='O'
        tfidata=fidatadefault[pdbdf.loc[i,'resname']][atype]
        fidatalist.append(tfidata)
    pdbdf['fidata']=fidatalist
    return

def fauchere(fi,pos,coords,max_dist2):
    '''
    this returns the contribution at each "pos" set of coordinates
    '''
    #calculated the squared distances
    dist2=((pos-coords)**2).sum(axis=1)
    #filter them by max dist squared
    filt=(dist2<=max_dist2)
    #dist2f=dist2[dist2<=max_dist2]
    #fif=fi[dist2<=max_dist2]
    if(filt.sum()>0):
        #for the valid ones sum the contribution
        return (100 * fi[filt] * np.exp(-np.sqrt(dist2[filt]))).sum()
        #return (100 * fi * np.exp(-np.sqrt(dist2))).sum()
    else:
        return 0.0

def getXYImage(fi,zi,coords,max_dist2,ss,sp,r,matrix):
    '''
    this sets the contribution at each z plane in the matrix
    '''
    matrix[zi]=[[fauchere(fi,np.array([ss[2]+r*k,ss[1]+r*j,ss[0]+r*zi]),coords,max_dist2)
                    for k in range(sp['x'])] for j in range(sp['y'])]
    print('plane',zi,'of',sp['z'])
    return 0

def getSpanStart(pdbdf,pad=8.0):
    '''
    this returns the start of the cube
    '''
    coords=pdbdf[['x','y','z']]
    lims=coords.describe().loc[['min','max']]
    lims.loc['min']-=pad
    lims.loc['max']+=pad
    return [lims.loc['min','z'],lims.loc['min','y'],lims.loc['min','x']]

def calcMLPMatrix(pdbdf,resolution=1.0,pad=8.0,max_dist=8.0,nthreads=16):
    '''
    calculates the lipiphilicity matrix via the fauchere method
    pdbdf should have an fidata column
    '''
    max_dist2=max_dist*max_dist
    print('max_dist^2',max_dist2)
    coords=pdbdf[['x','y','z']]
    lims=coords.describe().loc[['min','max']]
    lims.loc['min']-=pad
    lims.loc['max']+=pad
    span=lims.loc['max']-lims.loc['min']
    spanpix=(span/resolution).astype(int)+1
    spanstart=[lims.loc['min','z'],lims.loc['min','y'],lims.loc['min','x']]
    print('cube start (z,y,x):',spanstart)
    print('cube pixels:',[spanpix['z'],spanpix['y'],spanpix['x']])
    #matrix=np.zeros([spanpix['z'],spanpix['y'],spanpix['x']],dtype=float)
    ss=spanstart
    r=resolution
    sp=spanpix
    coords=coords.values
    fi=pdbdf['fidata'].values
    matrix=[None]*sp['z']
    que=[]
    st=time.time()
    with ThreadPoolExecutor(max_workers=nthreads) as executor:
        for i in range(sp['z']):
            fu=executor.submit(getXYImage,fi,i,coords,max_dist2,ss,sp,r,matrix)
            que.append(fu)
    for j in range(sp['z']):
        if(not que[j].done()):
            time.sleep(0.5)
    et=time.time()
    print('time',et-st)
    return np.array(matrix),spanstart

def interp3D(matrix,coords,span_start,resolution):
    '''
    this interpolates the matrix at the given coordinates given the span start and resolution
    '''
    shifted=coords-span_start
    scaled=shifted/resolution
    print('scaled',scaled)
    span_end=span_start+np.array(matrix.shape)*resolution
    #print('end',span_end)
    if(np.any(scaled<0.0)):
        return 0.0
    if(np.any(scaled>=(np.array(matrix.shape)-1))):
        return 0.0
    prevz=int(np.floor(shifted[0]))
    remz=scaled[0]-prevz
    z1=interp2D(matrix[prevz],scaled[2],scaled[1])
    z2=interp2D(matrix[prevz+1],scaled[2],scaled[1])
    return z1+remz*(z2-z1)

def interp2D(img,x,y):
    '''
    this interpolates the image at the given x,y
    '''
    prevx=int(np.floor(x))
    remx=x-prevx
    prevy=int(np.floor(y))
    remy=y-prevy
    ul=img[prevy,prevx]
    ur=img[prevy,prevx+1]
    ll=img[prevy+1,prevx]
    lr=img[prevy+1,prevx+1]
    yl=ul+remx*(ur-ul)
    yr=ll+remx*(lr-ll)
    return yl+remy*(yr-yl)

def getCoordsValues(coords,lmat,spanstart,resolution=1.0):
    '''
    returns the lipophilicity values from the matrix interpolated at the given coordinates
    the coords array is a list of z,y,x coordinates
    '''
    values=[]
    for i in range(len(coords)):
        val=interp3D(lmat,coords[i],spanstart,resolution)
        values.append(val)
    return values
