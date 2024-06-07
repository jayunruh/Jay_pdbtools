#this version doesn't have the plotly and py3Dmol dependencies (or functions)
import pandas as pd
#import plotly.graph_objects as go
#import plotly.express as px
import scipy.spatial as ss
import scipy.ndimage as ndi
import numpy as np
#import py3Dmol
import re
from io import StringIO
import subprocess

def findBadLine(lines):
    '''
    this finds the first line that can't be space delimited and returns that position
    examples are negative long format coordinates
    and residue numbers over 999
    '''
    fpos=-1
    for i in range(len(lines)):
        if(not (lines[i].startswith('ATOM') or lines[i].startswith('HETATM'))):
            continue
        m=re.search('\d-\d',lines[i])
        if(m):
            fpos=m.start()
            break
        #search for the chain letter followed by 4 digits
        m=re.search('[A-Za-z]\d{4}',lines[i])
        if(m):
            fpos=m.start()
            break
    return fpos

def fixPDB(fname):
    '''
    this converts a pdb file into a space delimitable string (can't be read by normal pdb readers)
    '''
    with open(fname,'r') as f:
        lines=f.readlines()
    ctr=0
    maxbadlines=100
    fpos=findBadLine(lines)
    while(fpos>0):
        #need to add a space at fpos for every line
        for i in range(len(lines)):
            lines[i]=lines[i][:(fpos+1)]+' '+lines[i][(fpos+1):]
        ctr+=1
        if(ctr>maxbadlines):
            break
        fpos=findBadLine(lines)
    return ''.join(lines)

def getpdbdf(fpath,skiprows=0,skipend=False):
    '''
    this is for pdb files with one model and no header
    for models with a header need to search for the initial row (don't do that for now)
    '''
    #read the file
    cnames=['type','atom','atype','resname','chain','residue','x','y','z','unk','temp','element']
    #note that the index_col=False here forces it not to use an index column
    sio=StringIO(fixPDB(fpath))
    pdbdf=pd.read_csv(sio,delim_whitespace=True,header=None,skiprows=skiprows,
                      names=cnames,on_bad_lines='skip',index_col=False)
    #label the columns
    #optionally eliminate the "end" and "ter" rows
    if(skipend):
        pdbdf=pdbdf.iloc[:-2]
    return pdbdf

def enforcedtypes(pdbdf):
    '''
    #this automatically forces atom, residue, x,y,z,temp into numeric data types
    '''
    pdbdf2=pdbdf.copy()
    #pdbdf2[['x','y','z','unk','temp']]=pd.to_numeric(pdbdf2[['x','y','z','unk','temp']],errors='coerce',downcast='float')
    pdbdf2['x']=pd.to_numeric(pdbdf2['x'],errors='coerce',downcast='float')
    pdbdf2['y']=pd.to_numeric(pdbdf2['y'],errors='coerce',downcast='float')
    pdbdf2['z']=pd.to_numeric(pdbdf2['z'],errors='coerce',downcast='float')
    pdbdf2['unk']=pd.to_numeric(pdbdf2['unk'],errors='coerce',downcast='float')
    pdbdf2['temp']=pd.to_numeric(pdbdf2['temp'],errors='coerce',downcast='float')
    #pdbdf2[['atom','residue']]=pd.to_numeric(pdbdf2[['atom','residue']],errors='coerce',downcast='integer')
    pdbdf2['atom']=pd.to_numeric(pdbdf2['atom'],errors='coerce',downcast='integer')
    pdbdf2['residue']=pd.to_numeric(pdbdf2['residue'],errors='coerce',downcast='integer')
    return pdbdf2

def cleanpdbdf(pdbdf):
    '''
    removes non-atom lines and enforces data types
    '''
    return enforcedtypes(pdbdf[pdbdf['type']=='ATOM'].reset_index(drop=True))

def writepdbdf(df,outpath=None,verbose=False):
    '''
    #this needs to be a space delimited aligned file whereonly x,y,z,unk, and temp are floating point
    #the coordinates have 3 decimals and unk and temp have 2
    #the atype is right justified unless it has >3 chars (necessary?)
    #there are two spaces to the far right
    '''
    #not sure if I need a TER row and an END row at the end
    cw=[4,7,5,4,2,4,12,8,8,6,6,12]

    def padInt(intval,totlen):
        fstr='{:'+str(totlen)+'}'
        return fstr.format(intval)

    def padFloat(fval,totlen,decimals):
        fstr='{:'+str(totlen)+'.'+str(decimals)+'f}'
        return fstr.format(fval)

    def padStr(sval,totlen,rightjust=False):
        if(not rightjust):
            fstr='{:'+str(totlen)+'}'
            return fstr.format(sval)
        else:
            fstr='{:>'+str(totlen)+'}'
            return fstr.format(sval)

    def writeRow(row):
        #the row order is type, atom, atype (right), resname, chain, residue, x,y,z, unk, temp, element
        outstr=padStr(row['type'],cw[0])+padInt(int(row['atom']),cw[1])+padStr(row['atype'],cw[2],True)
        outstr+=padStr(row['resname'],cw[3],True)+padStr(row['chain'],cw[4],True)+padInt(int(row['residue']),cw[5])
        outstr+=padFloat(row['x'],cw[6],3)+padFloat(row['y'],cw[7],3)+padFloat(row['z'],cw[8],3)
        outstr+=padFloat(row['unk'],cw[9],2)+padFloat(row['temp'],cw[10],2)+padStr(row['element'],cw[11],True)+'  '
        return outstr

    pdbstr=''
    for i in range(len(df)):
        tstr=writeRow(df.iloc[i])
        pdbstr+=tstr+'\n'
        if(verbose):
            print(tstr)
    lastrow=df.iloc[-1]
    tstr=padStr('TER',cw[0])+padInt(int(lastrow['atom'])+1,cw[1])+padStr(lastrow['resname'],cw[2]+cw[3],True)
    tstr+=padStr(lastrow['chain'],cw[4],True)+padInt(int(lastrow['residue']),cw[5])
    tstr+=''.join([' ']*54)
    pdbstr+=tstr+'\n'
    pdbstr+='END   '
    if(outpath is not None):
        with open(outpath,'w') as f:
            f.write(pdbstr)
    return pdbstr

def getFASTA(fpath):
    '''
    reads a fasta file and returns a data frame and a list of lines
    '''
    with open(fpath) as f:
        lines=f.readlines()

    #clean things up but removing comments and stripping out whitespace
    lines=[lines[i].strip() for i in range(len(lines)) if (not lines[i].startswith('#'))]
    currid=''
    currseq=''
    ids=[]
    seqs=[]
    for i in range(len(lines)):
        if(lines[i].startswith('>')):
            if(currid==''):
                currid=lines[i][1:]
            else:
                ids.append(currid)
                currid=lines[i][1:]
                seqs.append(currseq)
                currseq=''
        else:
            currseq+=lines[i]
    seqs.append(currseq)
    ids.append(currid)
    return pd.DataFrame({'id':ids,'sequence':seqs}),lines

def getFASTA2(lines):
    '''
    reads a FASTA file from it's lines and returns a dataframe
    '''
    #clean things up but removing comments and stripping out whitespace
    lines=[lines[i].strip() for i in range(len(lines)) if (not lines[i].startswith('#'))]
    currid=''
    currseq=''
    ids=[]
    seqs=[]
    for i in range(len(lines)):
        if(lines[i].startswith('>')):
            if(currid==''):
                currid=lines[i][1:]
            else:
                ids.append(currid)
                currid=lines[i][1:]
                seqs.append(currseq)
                currseq=''
        else:
            currseq+=lines[i]
    seqs.append(currseq)
    ids.append(currid)
    return pd.DataFrame({'id':ids,'sequence':seqs})

def writeA3M(fpath,wtid,wtseq,ids,seqs):
    '''
    #this writes an a3m file which is just a fasta with no line returns in the sequence
    '''
    with open(fpath,'w') as f:
        f.write('>'+wtid+'\n')
        f.write(wtseq+'\n\n')
        for i in range(len(ids)):
            f.write('>'+ids[i]+'\n')
            f.write(seqs[i]+'\n\n')
    return

def splitDF(df,splitcol,splitval):
    '''
    this splits a dataframe vertically like you would split a string horizontally
    splits happend at splitval instances in splitcol
    returns a list of dataframes
    '''
    splitpos=np.where(df[splitcol]==splitval)[0]
    splitpos=np.insert(splitpos,0,[0])
    splitpos=np.append(splitpos,len(df))
    #print(splitpos)
    return [df.iloc[(splitpos[i]+1):splitpos[i+1]] for i in range(len(splitpos)-1)]

def processDocking(pdbdf,keywords):
    '''
    #need to split up all of the ligands and calculate their centers of mass
    #try to pull out energy and deltaG for each one
    #the end of each ligand will be the TER type
    returns the target dataframe, ligand dataframes, centers of mass, and key values
    '''
    sdf=splitDF(pdbdf,'type','TER')
    targetdf=sdf[0]
    print('# ligands = '+str(len(sdf)-1))
    #each ligand will be preceeded by REMARK lines with details about the docking
    #the details are in the atom column and the values are in the atype column
    #all other coordinates are in the atom type but shifted one column left
    ligdfs=[]
    #headers=[]
    coms=[]
    keyvals=[]
    for i in range(1,len(sdf)):
        header=sdf[i][sdf[i]['type']!='ATOM']
        ligdf=sdf[i][sdf[i]['type']=='ATOM']
        ligdf.columns=ligdf.columns[:5].to_list()+ligdf.columns[6:].to_list()+['extra']
        ligdfs.append(ligdf)
        atoms=ligdf['atype'].str.slice(0,1)
        ligdfnoh=ligdf[atoms!='H']
        com=ligdfnoh.loc[:,['x','y','z']].values.mean(axis=0)
        coms.append(com)
        keyval=[]
        for j in range(len(keywords)):
            tkeyval=header[header['atom']==(keywords[j]+':')]['atype'].values
            keyval.append(tkeyval[0])
        keyvals.append(keyval)
    return targetdf,ligdfs,np.array(coms),np.array(keyvals)

#get just the backbones
def getBackbone(pdbdf):
    return pdbdf[(pdbdf['atype']=='N') | (pdbdf['atype']=='CA') | (pdbdf['atype']=='C')]

def getCA(pdbdf):
    return pdbdf[pdbdf['atype']=='CA']

def getSeq(cadf):
    '''
    #this get's the AA sequence from a c alpha pdb dataframe
    returns the sequence string and list of residues
    '''
    resnames=['GLY','ALA','VAL','ILE','LEU','MET','PHE','TYR','TRP','SER',
             'THR','ASN','GLN','CYS','PRO','ARG','HIS','LYS','ASP','GLU']
    resabbr=['G','A','V','I','L','M','F','Y','W','S',
             'T','N','Q','C','P','R','H','K','D','E']
    mapdict={resnames[i]:resabbr[i] for i in range(len(resnames))}
    reslist=cadf['resname'].values
    seq=[mapdict[reslist[i]] for i in range(len(reslist))]
    seqidx=[resnames.index(reslist[i]) for i in range(len(reslist))]
    return ''.join(seq),seqidx

def getPaddedStr(intval,padlen):
    padding=''.join([' ']*padlen)
    return (padding+str(intval))[-padlen:]

def writeNdx(fname,label,poss,ncols=15):
    lastpos=poss[-1]
    padlen=len(str(lastpos))+1
    with open(fname,'w') as f:
        f.write('[ '+label+' ]\n');
        col=0
        sb=''
        for i in range(len(poss)):
            sb=sb+getPaddedStr(poss[i],padlen)
            col+=1
            if(col>=ncols):
                f.write(sb[1:]+' \n')
                sb=''
                col=0
        f.write(sb[1:]+' \n')
    return

def getRMSD(pdbdf1,pdbdf2):
    #assume that the atom sets are identical here (e.g. a set of ca values would be easiest)
    return np.sqrt(((pdbdf2[['x','y','z']].values/10-pdbdf1[['x','y','z']].values/10)**2).sum(axis=1).mean())

def alignFullRMSD(pdbdf1,pdbdf2,aset='ca'):
    '''
    #this aligns two full atom sets together referencing 'ca' or 'backbone' or 'all' subsets
    #if reslist is not "None" that defines ths subset of atoms that are aligned
    '''
    subdf1=pdbdf1.copy()
    subdf2=pdbdf2.copy()
    if(aset=='ca'):
        subdf1=getCA(pdbdf1).reset_index(drop=True)
        subdf2=getCA(pdbdf2).reset_index(drop=True)
    elif(aset=='backbone'):
        subdf1=getBackbone(pdbdf1).reset_index(drop=True)
        subdf2=getBackbone(pdbdf2).reset_index(drop=True)
    #now truncate the dataframes to the same length
    if(len(subdf1)!=len(subdf2)):
        minlen=min(len(subdf1),len(subdf2))
        subdf1=subdf1.iloc[:minlen]
        subdf2=subdf2.iloc[:minlen]
    _,_,trans,_,com1,com2=alignRMSD(subdf1,subdf2)
    #now transform the original atom sets with the same transformation
    shiftdf1=pdbdf1.copy()
    shiftdf2=pdbdf2.copy()
    shiftdf1.loc[:,['x','y','z']]-=com1
    shiftdf2.loc[:,['x','y','z']]-=com2
    trans2=trans.apply(shiftdf2.loc[:,['x','y','z']]/10)
    shiftdf2.loc[:,['x','y','z']]=trans2*10
    return shiftdf1,shiftdf2

def transformpdbdf(pdbdf,com,trans=None,comdest=None):
    '''
    #this takes a center of mass and transformation from a subset
    #subtract the center of mass from the full data frame
    #then run the transformation
    '''
    shiftdf=pdbdf.copy()
    shiftdf.loc[:,['x','y','z']]-=com
    if(trans is not None):
        trans2=trans.apply(shiftdf.loc[:,['x','y','z']]/10)
        shiftdf.loc[:,['x','y','z']]=trans2*10
    if(comdest is not None):
        shiftdf.loc[:,['x','y','z']]+=comdest
    return shiftdf

def alignRMSD(pdbdf1,pdbdf2):
    '''
    #this aligns two identical sets of atoms to one another with best fit and returns the RMSD
    #assume that the atom sets are identical here
    #return aligned_df1,aligned_df2,transformation,rmsd,center1,center2
    '''
    com1=pdbdf1[['x','y','z']].mean(axis=0)
    com2=pdbdf2[['x','y','z']].mean(axis=0)
    #make copies with com at 0
    shiftdf1=pdbdf1.copy()
    shiftdf1.loc[:,['x','y','z']]-=com1
    shiftdf2=pdbdf2.copy()
    shiftdf2.loc[:,['x','y','z']]-=com2
    #now get the transformation matrix
    trans,rms=ss.transform.Rotation.align_vectors(shiftdf1.loc[:,['x','y','z']]/10,shiftdf2.loc[:,['x','y','z']]/10)
    trans2=trans.apply(shiftdf2.loc[:,['x','y','z']]/10)
    shiftdf2.loc[:,['x','y','z']]=trans2*10
    return shiftdf1,shiftdf2,trans,getRMSD(shiftdf1,shiftdf2),com1,com2

def alignShiftRMSD(pdbdf1,pdbdf2,minshift=-10,maxshift=10):
    '''
    attempts to align two pdbdfs of different length with offset shifts
    brute force search approach minimizing RMSD
    returns minrmsd, bestshift, aligned pdbdf1, and aligned pdbdf2
    '''
    minrmsd=np.inf
    bshift=minshift
    b1=None
    b2=None
    for shift in range(minshift,maxshift+1):
        if(shift<0):
            minlen=min(len(pdbdf1)+shift,len(pdbdf2))
            a1,a2,_,rmsd,_,_=alignRMSD(pdbdf1.iloc[-shift:(minlen-shift)],pdbdf2.iloc[:minlen])
            if(rmsd<minrmsd):
                minrmsd=rmsd
                bshift=minshift
                b1=a1
                b2=a2
        else:
            minlen=min(len(pdbdf1),len(pdbdf2)-shift)
            a1,a2,_,rmsd,_,_=alignRMSD(pdbdf1.iloc[:minlen],pdbdf2.iloc[shift:(minlen+shift)])
            if(rmsd<minrmsd):
                minrmsd=rmsd
                bshift=minshift
                b1=a1
                b2=a2
    return minrmsd,bshift,b1,b2

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
    coords=pdbdf[['x','y','z']]
    srad=vdwrad/resolution
    print('calibrated sphere radius:',srad)
    lims=coords.describe().loc[['min','max']]
    lims.loc['min']-=vdwrad
    lims.loc['max']+=vdwrad
    span=lims.loc['max']-lims.loc['min']
    spanpix=(span/resolution).astype(int)+1
    spanstart=[lims.loc['min','z'],lims.loc['min','y'],lims.loc['min','x']]
    print('cube start (z,y,x):',spanstart)
    print('cube pixels:',[spanpix['z'],spanpix['y'],spanpix['x']])
    mask=np.full([spanpix['z'],spanpix['y'],spanpix['x']],True,dtype=bool)
    zpos=np.floor((coords['z']-spanstart[0])/resolution).astype(int)
    ypos=np.floor((coords['y']-spanstart[1])/resolution).astype(int)
    xpos=np.floor((coords['x']-spanstart[2])/resolution).astype(int)
    mask[zpos,ypos,xpos]=False
    surf=ndi.distance_transform_edt(mask)<=srad
    return ndi.binary_fill_holes(surf),spanstart

def getSurfaceDistance(coords,surf,spanstart,resolution=0.25):
    '''
    gets the distance to a pre-calculated molecular surface which is a true/false array
    coords is a numpy array of z,y,x float arrays
    spanend=spanstart+surf.shape*resolution
    '''
    edt=ndi.distance_transform_edt(surf)
    dist=np.full(len(coords),1.0)
    #if the coordinates are outside the box, set the distance to 0
    coords2=((coords-spanstart)/resolution).astype(int)
    for i in range(len(coords2)):
        if(coords2.min()<0 or (coords2>=surf.shape).max()):
            dist[i]=np.NAN
        else:
            dist[i]=edt[coords2[i,0],coords2[i,1],coords2[i,2]]*resolution
    return dist

def inBox(coords,llims,ulims):
    for i in range(len(coords)):
        if(coords[i]<llims[i] or coords[i]>=ulims[i]):
            return False
    return True

def getAllDihedrals(pdbdf):
    #assume all one chain and no missing residues
    dihedrals=[]
    for residue in pdbdf['residue'].unique():
        dihedrals.append(getDihedrals(pdbdf,residue))
    return np.array(dihedrals)

def getDihedrals(pdbdf,resnum):
    #dihedrals are phi (prev_C, N, Ca, C)
    #psi: (N,Ca,C,next_N)
    #omega: (Ca,C,next_N,Next_Ca)
    maxres=pdbdf['residue'].max()
    if(resnum<1 or resnum>maxres):
        return None,None,None
    prevres=None
    if(resnum>1):
        prevres=getBackbone(pdbdf[pdbdf['residue']==(resnum-1)]).set_index('atype')
    res=getBackbone(pdbdf[pdbdf['residue']==resnum]).set_index('atype')
    nextres=None
    if(resnum<maxres):
        nextres=getBackbone(pdbdf[pdbdf['residue']==(resnum+1)]).set_index('atype')
    phi=None
    psi=None
    omega=None
    if(prevres):
        phicoords=[prevres.loc['C',['x','y','z']].values,
                res.loc['N',['x','y','z']].values,
                res.loc['CA',['x','y','z']].values,
                res.loc['C',['x','y','z']].values]
        phi,_,_=getDihedral(phicoords)
        phi=-phi
    if(nextres):
        psicoords=[res.loc['N',['x','y','z']].values,
                res.loc['CA',['x','y','z']].values,
                res.loc['C',['x','y','z']].values,
                nextres.loc['N',['x','y','z']].value]
        psi,_,_=getDihedral(psicoords)
        omegacoords=[res.loc['CA',['x','y','z']].values,
                res.loc['C',['x','y','z']].values,
                nextres.loc['N',['x','y','z']].values,
                nextres.loc['CA',['x','y','z']].value]
        omega=getDihedral(omegacoords)
    return phi,psi,omega

def getDihedral(coords):
    #this gets the dihedral angle for a set of 4 coordinates
    #first get the normals for the two sets of 3 coordinates
    #the normals are given by the cross product of the neighboring vectors
    vec1=coords[0]-coords[1]
    vec2=coords[2]-coords[1]
    norm1=np.cross(vec1,vec2)
    print('norm1',norm1)
    vec3=coords[1]-coords[2]
    vec4=coords[3]-coords[2]
    norm2=np.cross(vec3,vec4)
    print('norm2',norm2)
    len1=np.sqrt((norm1**2).sum())
    len2=np.sqrt((norm2**2).sum())
    #now get the angle between the normals (dot-prod/length product)
    return np.arccos(np.dot(norm1,norm2)/(len1*len2)),norm1,norm2

def getSecondaryStructures(dihedrals):
    #here we estimate the secondary structures from the dihedrals (fairly poor accuracy)
    ssdict={
        'alpha-helix':{'symbol':'a','phi':[-3.4906585, 0],'psi':[-2.0943951, 0.698132]},
        'polyproline-2':{'symbol':'p','phi':[-1.57079633, 0],'psi':[0.698132, 4.18879]},
        'beta-sheet':{'symbol':'b','phi':[-3.49066, -1.57079633],'psi':[0.698132, 4.18879]},
        'left-handed':{'symbol':'l','phi':[0, 2.79253],'psi':[-1.57079633, 1.91986]},
        'gamma':{'symbol':'g','phi':[0, 2.79253],'psi':[1.91986, 4.71239]},
        'cis':{'symbol':'c','omega':[-1.57079633, 1.57079633]}
    }
    sssymbols={ssdict[ss]['symbol']:ss for ss in ssdict}
    sssymbols['d']='disorder'
    sss=[]
    for dihedral in dihedrals:
        thisss=None
        for ss in ssdict:
            ss2=ssdict[ss]
            if(dihedral[0] and dihedral[1] and dihedral[2]):
                if('omega' in ss2 and (ss2['omega'][0]<=dihedral[2]<=ss2['omega'][1])):
                    #this is the cis structure
                    thisss=ss2['symbol']
                    break
                elif((ss2['phi'][0]<=dihedral[0]<=ss2['phi'][1]) and (ss2['psi'][0]<=dihedral[0]<=ss2['psi'][1])):
                    thisss=ss2['symbol']
                    break
        if(not thisss):
            thisss='d'
        sss.append(thisss)
    ssnames=[sssymbols[sss[i]] if sss[i] else None for i in range(len(sss))]
    return sss,ssnames

#this is the path for the stride program to calculate secondary structures
stridepath='/n/projects/jru/public/vmd-1.9.3/lib/stride/stride_LINUXAMD64'

def runPipe(cmd):
    '''
    This runs a subprocess pipe and returns the stdout and stderr as lists of lines
    '''
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE, shell=True)
    (out,err)=proc.communicate()
    out=out.decode('utf-8')
    err=err.decode('utf-8')
    return out.split('\n'),err.split('\n')

def getStrideSS(pdbpath):
    '''
    uses the binary program stride shipped with vmd to get secondary structures from a pdb file
    set the stridepath variable to the location of the stride binary on your computer (often in the VMD install)
    returns the secondary structure list and the sequence
    '''
    strideout,_=runPipe(stridepath+' -o '+pdbpath)
    #all lines start with a line type and end with ~~~~
    #the lines starting with SEQ have the sequence
    #the lines with ss start with STR
    #the lines are space aligned from the 10th position to the 60th
    seqlines=[strideout[i][10:60] for i in range(len(strideout)) if strideout[i].startswith('SEQ')]
    sslines=[strideout[i][10:60] for i in range(len(strideout)) if strideout[i].startswith('STR')]
    #the lines are space aligned from the 10th position to the 60th
    seq=''.join(seqlines)
    ss=''.join(sslines)
    ss=ss.replace(' ','C')
    #need to remove the trailing spaces as defined by the seq record
    seq=seq.strip()
    return ss[:len(seq)],seq

def calculate_lDDT(pdbdf1, pdbdf2):
    '''
    this calculates the lDDT (local distance difference test) score for two (cleaned) pdb structures
    this metric is translation and rotation invariant (Mariani et al., Bioinformatics, 2013)
    pdbdf1 and pdbdf2 are the two dataframes of the two structures
    pdbdf1 and pdbdf2 should have the same number of rows (atoms)
    returns a dataframe with the lDDT score for each atom and the total number of measured distances (over 4 tolerances)
    '''
    #remove hydrogens
    pdbdf1nh=pdbdf1[pdbdf1['element']!='H']
    pdbdf2nh=pdbdf2[pdbdf2['element']!='H']
    residues=pdbdf1['residue'].unique()
    lDDT_scores=[]
    tot_distss=[]
    resnames=[]
    for residue in residues:
        #now calculate the lDDT score for each residue
        tlddt,tot_dists=getResiduelDDT(pdbdf1nh,pdbdf2nh,residue)
        resname=pdbdf1nh[pdbdf1nh['residue']==residue].iloc[0]['resname']
        lDDT_scores.append(tlddt)
        resnames.append(resname)
        tot_distss.append(tot_dists)
    outdf=pd.DataFrame({'residue':residues,'resname':resnames,'lddt':lDDT_scores,'total_distances':tot_distss})
    return outdf

def getResiduelDDT(pdbdf1,pdbdf2,resnum):
    '''
    calculates getAtomlDDT over a pair of residues
    when atom numbers don't match, try to use atom type instead
    '''
    subdf1=pdbdf1[pdbdf1['residue']==resnum]
    subdf2=pdbdf2[pdbdf2['residue']==resnum]
    if(len(subdf2)<1):
        return np.nan
    lddt_scores=[]
    tot_dists=[]
    for i in range(len(subdf1)):
        atomnum=subdf1.iloc[i]['atom']
        atomnum2=subdf2.iloc[i]['atom']
        if(atomnum!=atomnum2):
            #if atom numbers don't match, try types
            temp=subdf2[subdf2['atype']==subdf1.iloc[i]['atype']]
            if(len(temp)==1):
                atomnum2=temp.iloc[0]['atom']
                lddt,inc_tot=getAtomlDDT(pdbdf1,pdbdf2,atomnum,atomnum2)
                lddt_scores.append(lddt)
                tot_dists.append(inc_tot)
        else:
            lddt,inc_tot=getAtomlDDT(pdbdf1,pdbdf2,atomnum,atomnum2)
            lddt_scores.append(lddt)
            tot_dists.append(inc_tot)
    return np.array(lddt_scores).mean(),np.array(tot_dists).sum()*4

def getAtomlDDT(pdbdf1,pdbdf2,atomnum1,atomnum2,tolerances=[0.5,1.0,2.0,4.0],inclusion_radius=15):
    '''
    This function returns the fraction of atoms the same distance from the atom of interest within a tolerance and within an inclusion radius
    pdbdf1 is the dataframe of the structure of interest
    pdbdf2 is the dataframe of the structure to compare
    atomnum is the atom number of the atom of interest
    tolerances are the tolerances for the distance difference between the two structures
    inclusion_radius is the radius within which to consider atoms
    '''
    atom1 = pdbdf1[pdbdf1['atom']==atomnum1]
    atom2 = pdbdf2[pdbdf2['atom']==atomnum2]
    res1=atom1['residue'].values[0]
    atom1_coords = atom1[['x','y','z']].values
    atom2_coords = atom2[['x','y','z']].values
    inclusion_radius_squared = inclusion_radius**2
    #get the squared distance between all atoms not in this residue and the selected atom
    dists1_squared=((pdbdf1[pdbdf1['residue']!=res1][['x','y','z']].values-atom1_coords)**2).sum(axis=1)
    dists2_squared=((pdbdf2[pdbdf2['residue']!=res1][['x','y','z']].values-atom2_coords)**2).sum(axis=1)
    #find which atoms are in the inclusion radius of either model
    inclusion_idx=(dists1_squared<inclusion_radius_squared) | (dists2_squared<inclusion_radius_squared)
    inc_tot=inclusion_idx.sum()
    dists1_squared=dists1_squared[inclusion_idx]
    dists2_squared=dists2_squared[inclusion_idx]
    distdiff=np.abs(np.sqrt(dists1_squared)-np.sqrt(dists2_squared))
    #now find the fraction of distances that are within tolerance
    #idx1=(dists1_squared<=inclusion_radius_squared)
    #idx2=(dists2_squared<=inclusion_radius_squared)
    #ftoler=np.array([float(((distdiff<tolerance) & idx1 & idx2).sum()) for tolerance in tolerances])
    ftoler=np.array([float(((distdiff<tolerance)).sum()) for tolerance in tolerances])
    ftoler=ftoler/float(inc_tot)
    return ftoler.mean()*100.0,inc_tot

def getDNANSeq(pdbdf):
    '''
    return the DNA pdbdf with attachment nitrogens (N9 for A G and N1 for T C)
    also return the DNA sequence
    '''
    #get all the combinatorial options for attachment nitrogens
    isAn=(pdbdf['atype']=='N9') & (pdbdf['resname']=='DA')
    isGn=(pdbdf['atype']=='N9') & (pdbdf['resname']=='DG')
    isTn=(pdbdf['atype']=='N1') & (pdbdf['resname']=='DT')
    isCn=(pdbdf['atype']=='N1') & (pdbdf['resname']=='DC')
    nadf=pdbdf[isAn | isGn | isTn | isCn]
    seq=''.join(nadf['resname'].str[1])
    return nadf,seq
