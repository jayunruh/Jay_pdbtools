#this is a relatively simple implementation of a protein hydrogen bond finder based on angle and distance
#note that hydrogens must be added prior to running this--I recommend gromacs for that task
import numpy as np
import pandas as pd

#here is the full list of potential residue, hydrogen, and donor atom names (not including backbone)
donor_list=[['TYR','HH','OH'],['TRP','HE1','NE1'],['SER','HG','OG'],['THR','HG1','OG1'],['ASN','HD21','ND2'],
            ['ASN','HD22','ND2'],['GLN','HE21','NE2'],['GLN','HE22','NE2'],['CYS','HG','SG'],['ARG','HE','NE'],
            ['ARG','HH11','NH1'],['ARG','HH12','NH1'],['ARG','HH21','NH2'],['ARG','HH22','NH2'],
            ['HIS','HE2','NE2'],['HIS','HD1','ND1'],['LYS','HZ1','NZ'],['LYS','HZ2','NZ'],['LYS','HZ3','NZ']]
# here's the full list of potential acceptor residue and atom names (not including backbone)
acceptor_list=[['ASN','OD1'],['GLN','OE1'],['ASP','OD2'],['ASP','OD1'],['GLU','OE1'],['GLU','OE2']]

def getMinDist(clist1,clist2):
    '''
    get the minimum distance between two lists of coordinates (e.g. from two residues)
    '''
    dist2=[((clist1-clist2[i])**2).sum(axis=1).min() for i in range(len(clist2))]
    return np.sqrt(np.array(dist2).min())

def getDistProfile(pdbdfa,pdbdfb):
    '''
    get the minimum distance for each residue in chain A to chain B
    '''
    resvalsa=pdbdfa['residue'].unique()
    resvalsb=pdbdfb['residue'].unique()
    distmat=[]
    for resa in resvalsa:
        #print('measuring distances to',resa)
        coordsa=pdbdfa[pdbdfa['residue']==resa][['z','y','x']].values
        distprof=[]
        for resb in resvalsb:
            coordsb=pdbdfb[pdbdfb['residue']==resb][['z','y','x']].values
            distprof.append(getMinDist(coordsa,coordsb))
        distmat.append(distprof)
    return distmat,resvalsa,resvalsb

def getContactResidues(pdbdf,cutoff=4.0,chain1='A',chain2='B'):
    '''
    get the list of residues that contact one another (dist<=cutoff) from chain1 and chain2
    '''
    pdbdfa=pdbdf[pdbdf['chain']==chain1]
    pdbdfb=pdbdf[pdbdf['chain']==chain2]
    distmat,resa,resb=getDistProfile(pdbdfa,pdbdfb)
    acont=[]
    bcont=[]
    for i in range(len(distmat)):
        for j in range(len(distmat[i])):
            if(distmat[i][j]<=cutoff):
                print('contact between',resa[i],resb[j])
                acont.append(resa[i])
                bcont.append(resb[j])
    acont=np.unique(acont)
    bcont=np.unique(bcont)
    acontdfs=[]
    bcontdfs=[]
    for i in range(len(acont)):
        acontdfs.append(pdbdfa[pdbdfa['residue']==acont[i]])
    for i in range(len(bcont)):
        bcontdfs.append(pdbdfb[pdbdfb['residue']==bcont[i]])
    return pd.concat(acontdfs),pd.concat(bcontdfs)

def getDALists(pdbdf):
    '''
    get the list of potential donor pairs (H and donor) and acceptors
    the donor pairs are returned as a list of pair lists
    the acceptor pairs are returned as a dataframe
    '''
    #go through the pdb and identify all of the donors and acceptors
    donordfs=[]
    #start with the backbone
    residues=pdbdf['residue'].unique()
    chains=pdbdf['chain'].unique()
    for chain in chains:
        for residue in residues:
            pair=[pdbdf[(pdbdf['residue']==residue) & (pdbdf['chain']==chain) & (pdbdf['atype']=='H')],
                    pdbdf[(pdbdf['residue']==residue) & (pdbdf['chain']==chain) & (pdbdf['atype']=='N')]]
            if(len(pair[0])>0 and len(pair[1])>0):
                donordfs.append(pair)
    #and now the donor sidechains
    for i in range(len(donor_list)):
        dseldf1=pdbdf[(pdbdf['resname']==donor_list[i][0]) & (pdbdf['atype']==donor_list[i][1])]
        for j in range(len(dseldf1)):
            tres=dseldf1.iloc[j]['residue']
            tchain=dseldf1.iloc[j]['chain']
            #print('donor found at residue',tres,'in chain',tchain)
            pair=[pd.DataFrame(dseldf1.iloc[j]).transpose(),
                pdbdf[(pdbdf['residue']==tres) & (pdbdf['chain']==tchain) & (pdbdf['atype']==donor_list[i][2])]]
            donordfs.append(pair)
    print(len(donordfs),'donor atoms found')
    #now the acceptor backbone
    acceptordf=pdbdf[pdbdf['atype']=='O']
    #and the acceptor side chains
    acceptordfs=[]
    for i in range(len(acceptor_list)):
        aseldf=pdbdf[(pdbdf['resname']==acceptor_list[i][0]) & (pdbdf['atype']==acceptor_list[i][1])]
        if(len(aseldf)>0):
            #print(len(aseldf),'acceptors found')
            acceptordfs.append(aseldf)
    if(len(acceptordfs)>0):
        acceptordf=pd.concat([acceptordf,pd.concat(acceptordfs)])
    print(len(acceptordf),'acceptor atoms found')
    return donordfs,acceptordf

def getVecAngle(c1,c2,c3):
    '''
    returns the angle c1-c2-c3
    '''
    v1=c1-c2
    v1n=v1/np.linalg.norm(v1)
    v2=c3-c2
    v2n=v2/np.linalg.norm(v2)
    return np.arccos(np.dot(v1n,v2n))

def getHBonds(acceptordf,donordfs,cutoff=4.0,maxangle=60.0):
    '''
    this takes a dataframe of acceptors and a list of donor and h pairs
    returns the hbond triples (dataframe rows), distances, and angles
    '''
    #now go through the acceptors and find the neighboring donors
    triples=[]
    dists=[]
    angles=[]
    for i in range(len(acceptordf)):
        arow=acceptordf.iloc[i]
        acoords=arow[['z','y','x']].values
        #measure the distance to the donor
        for j in range(len(donordfs)):
            dcoords=donordfs[j][1].iloc[0][['z','y','x']].values
            dhcoords=donordfs[j][0].iloc[0][['z','y','x']].values
            dist=np.sqrt(((acoords-dcoords)**2).sum())
            if(dist<=cutoff):
                #now check the angle
                angle=180-np.degrees(getVecAngle(acoords,dhcoords,dcoords))
                if(angle<=maxangle):
                    print('hbond found at acceptor',arow['residue'],'donor',donordfs[j][1]['residue'].values[0])
                    dists.append(dist)
                    angles.append(angle)
                    tdf=pd.concat([donordfs[j][0],donordfs[j][1],pd.DataFrame(arow).transpose()])
                    triples.append(tdf)
    return triples,dists,angles

def getHBondDF(triples,dists,angles):
    '''
    makes a list of hydrogen bond donor, acceptor, distance, and angle metrics as a data frame
    '''
    hbdicts=[]
    for i in range(len(triples)):
        dr=triples[i].iloc[1]
        ar=triples[i].iloc[2]
        hbdict={'dchain':dr['chain'],'dresidue':dr['residue'],'dresname':dr['resname'],'dtype':dr['atype'],
                'dhtype':triples[i].iloc[0]['atype'],'achain':ar['chain'],'aresidue':ar['residue'],
                'aresname':ar['resname'],'atype':ar['atype'],'dist':dists[i],'angle':angles[i]}
        hbdicts.append(hbdict)
    return pd.DataFrame(hbdicts)
