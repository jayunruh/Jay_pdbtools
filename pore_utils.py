import numpy as np
import scipy.ndimage as ndi

def buildCirc(shape,centerx,centery,radius):
    '''
    builds a boolean circle center and radius
    '''
    width=shape[1]
    height=shape[0]
    xg,yg=np.meshgrid(range(width),range(height))
    rad2=radius*radius
    circ=((xg-centerx)**2+(yg-centery)**2)>rad2
    return circ

def buildCylinder(shape,centerx,centerz,radius):
    '''
    builds a boolean cylinder in y with center and radius in xz
    build a planar circle and repeat it
    '''
    circ=buildCirc((shape[0],shape[2]),centerx,centerz,radius)
    cyl=np.repeat(circ[:,np.newaxis,:],shape[1],axis=1)
    return cyl

def filterPore(surf,ymin=25,ymax=375,xzc=215,xzw=110):
    '''
    #now create a "pore" object by cropping this between ymin and ymax
    #radially fill outside xzc +/- xzw
    '''
    pore=(~surf).copy()
    pore[:,:ymin,:]=0
    pore[:,ymax:,:]=0
    cyl=buildCylinder(pore.shape,xzc,xzc,xzw)
    pore[cyl]=0
    return pore

def getPoreProfile(pore):
    '''
    measure the minimum diameter profile of a pore aligned in y
    treat the minimum as the closest distance to the centroid
    '''
    xg,zg=np.meshgrid(range(pore.shape[2]),range(pore.shape[0]))
    distprofile=[]
    aprofile=[]
    centprofile=[]
    for i in range(pore.shape[1]):
        #start by getting the largest pore object
        slice=pore[:,i,:]
        if(slice.max()<1):
            distprofile.append(None)
            centprofile.append(None)
            aprofile.append(None)
            continue
        obj,nobj=ndi.label(slice)
        centroids=ndi.center_of_mass(slice,labels=obj,index=range(1,nobj+1))
        areas=ndi.sum(slice,labels=obj,index=range(1,nobj+1))
        if(nobj>0):
            selobj=np.argmax(areas)+1
            obj=obj[obj==selobj]
        else:
            selobj=1
        aprofile.append(areas[selobj-1])
        centprofile.append(centroids[selobj-1])
        dist2=(xg-centprofile[i][1])**2+(zg-centprofile[i][0])**2
        mindist=np.sqrt(dist2[~slice].min())
        distprofile.append(mindist)
        #print(i,mindist)
    return distprofile,aprofile,centprofile

def getPoreProfile2(pore,centerx,centerz):
    '''
    measure the minimum diameter profile of a pore aligned in y
    treat the minimum as the closest distance to the centerx, centerz position
    '''
    xg,zg=np.meshgrid(range(pore.shape[2]),range(pore.shape[0]))
    dist2=(xg-centerx)**2+(zg-centerz)**2
    distprofile=[]
    for i in range(pore.shape[1]):
        slice=pore[:,i,:]
        if(slice.max()<1):
            distprofile.append(np.nan)
            continue
        mindist=np.sqrt(dist2[~slice].min())
        distprofile.append(mindist)
        #print(i,mindist)
    return distprofile
