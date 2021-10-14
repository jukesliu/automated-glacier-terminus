#!/usr/bin/env python
# coding: utf-8

# In[2]:


def generate_circle_image(r, imagex, imagey):
    from PIL import Image, ImageDraw
    # Generate circle with radius r in center of an image with dimensions imagex by imagey
    x = int(imagex/2); y=int(imagey/2)
    image = Image.new(mode="L", size=(imagex, imagey))
    draw = ImageDraw.Draw(image)
    leftUpPoint = (x-r, y-r); rightDownPoint = (x+r, y+r)
    draw.ellipse([leftUpPoint, rightDownPoint], fill=(255)) # draw white ellipse
    return image


# In[4]:


# Forward and inverse DFT functinos
def ft2(f, delta_f):
    import numpy as np
    import pyfftw
    
    if len(locals()) < 2: # if no spacing specified
        delta_f = 1 # default spacing = 1
    
    pyfftwobj = pyfftw.builders.fft2(np.fft.fftshift(f)); pyfftwobj = pyfftwobj();
    F = np.fft.fftshift(pyfftwobj)* (delta_f**2)
    
    if F.shape[0] != F.shape[1]: # if not square:
        F = np.transpose(F) # transpose result for unequal image dimensions
                    
    return F


# In[5]:


def ift2(F, delta_f):
    import numpy as np
    import pyfftw
    
    if len(locals()) < 2: # if no spacing specified
        delta_f = 1 # default spacing = 1
    
    N = np.max(F.shape)
    ipyfftwobj = pyfftw.builders.ifft2(np.fft.ifftshift(F)); ipyfftwobj = ipyfftwobj();
    f = np.fft.ifftshift(ipyfftwobj)*(N*delta_f)**2
    
    if f.shape[0] != f.shape[1]: # if not square:
        f = np.transpose(f) # transpose result for unequal image dimensions
    
    return f


# In[6]:

def recon(arr_shape, arr_raveled):
# Reconstruct an array that has been raveled in column-major
# INPUTS:
# - arr_shape: shape tuple for the original array (output from arr.shape)
# - arr_raveled: the column-major (order ='F', fortran style) raveled array
# OUTPUTS:
# - arr_recon: reconstructed array
    import numpy as np

    arr_recon = np.zeros(arr_shape) # intialize the reconstructed array
    for l in np.arange(0, arr_shape[1]): # for each column
        linidx = np.arange(0,arr_shape[0])+(l*arr_shape[0]) # grab the linear indices for the column
        arr_recon[:,l] = arr_raveled[linidx] # grab the raveled array values and enter into column
    return arr_recon


# maxima interpolation
def minterp(lx,ly,dx_norm,dy_norm,i):
    import numpy as np
    # bilinear interpolation
    x_interp = np.mod(i,lx) + dx_norm[i+1]
    y_interp = np.floor(i/lx) + dy_norm[i+1]
    
    return x_interp, y_interp      


# In[1]:

def pad_square(nparray):
    import numpy as np
    
    # Pad an array to be square if not square
    # INPUT: the array
    # OUTPUT: the square (padded) array
    [ly,lx] = nparray.shape
    if lx > ly:
        zeros = np.zeros((lx,lx))
        zeros[:ly,:lx] = nparray
    elif ly > lx:
        zeros = np.zeros((ly,ly))
        zeros[:ly,:lx] = nparray
    else:
        zeros = nparray
    nparray = zeros
    
    return zeros


# In[1]:

def wtmm2d(I,wavelet,scale):
    import numpy as np
    from ttictoc import tic,toc
    
    # Perform 2D Wavelet Transform Modulus Maxima (WTMM) Method on image I
    # with wavelet at a user-specified scale.
    #
    # Input:
    # I       = image (numpy array)
    # wavelet = name of wavelet to use
    # scale   = scale of transform
    #
    # Output:
    #   mm = moduli maxima
    #   m  = moduli
    #   a  = arguments
    #
    # Author: Jukes Liu (translated from MATLAB code by Zach Marin, based on code by Pierre Kestener)
    #
    # Modifications made by Andre Khalil following the function
    # Remove_Gradient_NonMaxima_Slice_2D in Xsmurf 
    # (Found in {Xsmurf-folder}/edge/extrema_core.c)
    #
    # As of now, this function (wtmm2d) seems to work as it should.
    # However, it only produces correct results for images of size 2^k.
    # There's a padding step that is probably missing in the FFT functions.
    #
    # Last Modified: 2021 10 03
    
#     if len(locals()) != 3: # if the number of args is < 3
#         print('Incorrect number of arguments. Syntax is [dx, dy, mm, m, a] = wtmm2d(I, wavelet, scale).')
     
#     tic()
    I = pad_square(np.array(I))
    
#     print(I.shape)
    
    # calculate spatial frequencies based on scale
    maxdim = I.shape[0] # either dimension is the maximum whne square
    delta = 1 
    K = 1/(maxdim*delta)
    fX = np.arange(-maxdim/2,maxdim/2,1)*K*scale
    [fx,fy] = np.meshgrid(fX,fX)
    
    # set wavelet based on input
    if wavelet == 'gauss':
        p = 1; sigmax = 1; sigmay = 1;
        psi = np.exp(-((sigmax*(fx))**2 + (sigmay*fy)**2)/2)
    elif wavelet == 'igauss':
        p = 1; sigmax = 1; sigmay = 1;
        psi = 1-np.exp(-((sigmax*(fx))**2 + (sigmay*fy)**2)/2)
        psi[psi==1] = 0
    elif wavelet == 'mexh':
        p = 2; sigmax = 1; sigmay = 1;
        psi = -(2*np.pi*(fx**2 + fy**2)**(p/2))*np.exp(-((sigmax*(fx))**2 + (sigmay*fy)**2)/2)
    else:
        print('ERROR: "'+wavelet+'" is not a valid wavelet. Options are "gauss", "igauss", and "mexh".') 
    
    # FFT the image
    F = ft2(I, delta);
    
    f = np.multiply(F,psi) # convolve image with the wavelet
    
    # create derivative approximators
    gx = 1j*2*np.pi*fx
    gy = 1j*2*np.pi*fy
        
    # numerical aproximations of derivatives and in-place inverse FT
    dy = np.real(ift2(np.multiply(gx,f), delta))
    dx = np.real(ift2(np.multiply(gy,f), delta))
    
    # Calculate modulus and argument from dx and dy (WTMM method)
    m = np.sqrt(dx**2 + dy**2)
    a = np.arctan2(dy,dx)
#     print(toc())
      
    # FIND MAXIMA USING INTERPOLATION OF M:
#     tic()
    # Remove non-maxima to find edges
    err_norm  = 0.0000005 # Epsilon value to find zeros. If the modulus is less than err_norm, set it to zero.
    err_deriv = 0.9995 # Epsilon value to determine interpolation type. 
    # If the derivative's absolute value is greater than err_deriv, use the nearest modulus value. 
    # Otherwise, perform a bi- or tri-linear interpolation.
    
    # compute normalized derivative WT values
    dx_norm = np.divide(dx,m)
    dy_norm = np.divide(dy,m)
    
    [ly,lx] = m.shape # grab lx and ly (now square)
    
    # imitate MATLAB looping order, along columns
    mshape = m.shape # store shape for reconstruction of original matrix
    m = np.ravel(m, order='F')
    dy_norm = np.ravel(dy_norm, order='F')
    dx_norm = np.ravel(dx_norm, order='F')
    mm = np.zeros(m.shape) # initialize mm
    
    idxs = np.arange(0, lx*ly) # vector of idxs for m
    
    # Modulus interpolation loop 
    for i in np.arange(0,lx*ly,1):
        # remove edges
        if (np.mod(i,lx) == 0) or (np.mod(i,lx) == lx-1) or (np.floor(i/lx) == 0) or (np.floor(i/lx) == ly-1):
            continue
        
        # remove small moduli
        if m[i] < err_norm:
            continue
        
#         print(np.mod(i,lx), np.floor(i/lx))
            
        # check if the nearest modulus value works
        if (np.abs(dx_norm[i]) > err_deriv) or (np.abs(dy_norm[i]) > err_deriv):
            xul = np.mod(i,lx) + dx_norm[i] + 0.5
            yul = np.floor(i/lx) + dy_norm[i] + 0.5
            
#             print(xul, yul)
                       
            if m[i] <= m[int(np.fix(xul) + np.fix(yul)*lx)]:
                continue

            xul = np.mod(i,lx) - dx_norm[i] + 0.5
            yul = np.floor(i/lx) - dy_norm[i] + 0.5
                       
            if m[i] < m[int(np.fix(xul) + np.fix(yul)*lx)]:
                continue

            # found a modulus maxima
            mm[i] = m[i]               

        # since it didn't, use a bilinear interpolation
        x_interp = np.mod(i,lx) + dx_norm[i]
        y_interp = np.floor(i/lx) + dy_norm[i]

        # make sure we are inside the image:
        if (x_interp < 0) or (x_interp >= lx-1) or (y_interp < 0) or (y_interp >= ly-1):
            continue
        
        # calculate shifts
        sx = x_interp - np.fix(x_interp) # fix rounds it towards 0
        sy = y_interp - np.fix(y_interp)
        sxsy = sx*sy
        
        # calculate bilinear interpolation coefficients
        c00 = 1-sx-sy+sxsy;
        c10 = sx-sxsy;
        c01 = sy-sxsy;
        c11 = sxsy;
        
        # We compare the modulus of the point with the
        # interpolated modulus. It must be larger to be
        # still considered as a potential gradient extrema.
        
        # Here, we consider that it is strictly superior.
        # The next comparison will be superior or equal.
        # This way, the extrema is in the light part of the image.
        # By inverting both tests, we can put it in the
        # dark side of the image.
        ul = int(np.fix(x_interp) + np.fix(y_interp)*lx)
        m_interp = m[ul]*c00 + m[ul+1]*c10 + m[ul+lx]*c01 + m[ul+lx+1]*c11 # +1 removed from indexing
        
#         print(ul)
        
        if m[i] <= m_interp:
            continue
        
        # Second point interpolated (notice sign switch to subtracting. dx_norm)
        x_interp = np.mod(i,lx) - dx_norm[i]
        y_interp = np.floor(i/lx) - dy_norm[i]
        
        # make sure we are inside the image:
        if (x_interp < 0) or (x_interp >= lx-1) or (y_interp < 0) or (y_interp >= ly-1):
            continue
    
        ul = int(np.fix(x_interp) + np.fix(y_interp)*lx)
        m_interp = m[ul]*c11 + m[ul+1]*c01 + m[ul+lx]*c10 + m[ul+lx+1]*c00 # different coefficients
        
        if m[i] < m_interp:
            continue
        
        mm[i] = m[i] # found a modulus maxima
    
    # reshape mm and m back into ly by lx
    m = recon(mshape, m)
    mm = recon(mshape,mm)
    
    dx = dx[:ly,:lx]; dy = dy[:ly,:lx]
    mm = mm[:ly,:lx]; m = m[:ly,:lx]; a = a[:ly,:lx]

#     print(toc())
    return dx, dy, mm, m, a

# In[9]:


def emask(box_array, var):
    import numpy as np
# Mask the outputs of wtmm2d using a binary mask of the terminus box.
# INPUTS:
#     - box_array: binary mask for box
#     - var: the variable matrix to mask (e.g. mm, a)
# OUTPUTS:
#     - maskedvar: the masked variable matrix

    box_array = pad_square(box_array) # pad the box_array to be square if not square

    if box_array.shape == var.shape: # check that the dimensions are the same
        maskedvar = np.multiply(box_array, var) # multiply the two elementwise           
        return maskedvar
    else:
        print('Dimemsions do not match:', box_array.shape, var.shape)


# In[10]:


def flip_search(searched0):
    import numpy as np
# Turns 0s into 1s and 1s into 0s
    search_flip = searched0
    
    for k in range(0, len(searched0)):
        val = searched0[k]
        if val == 0: # if it's 0
            search_flip[k] = 1 # replace with 1
        else: # if it's 1
            search_flip[k] = 0 # replace with 0
            
    return np.transpose(search_flip) # transpose


# In[11]:


def wtmmchains(mm, a, keepClosed, scale):
# Chains the modulus maxima from the 2D WTMM.
# Searches for 8 neighboring mms.
# INPUTS:
#    - mm: the modulus maxima matrix from wtmm2d
#    - a: the argument matrix from wtmm2d
#    - keepClosed: 1 to keep the closed loops, 0 to remove them
#    - scale: scale of analysis (float)
# OUTPUTS:
#    - cmm: a list of chain objects corresponding to the chained modulus maxima

    import numpy as np
    
    # define the chain object class with the following parameters:
    class chain:
        def __init__(self, size, linemeanmod, mass, scaledmass, args, ix, iy):
            self.size = size
            self.linemeanmod = linemeanmod
            self.mass = mass
            self.scaledmass = scaledmass
            self.args = args
            self.ix = ix
            self.iy = iy

        def show(self):
            print('Size:', self.size)
            print('Linemeanmod:', self.linemeanmod)
            print('Mass:', self.mass)
            print('Metric:', self.metric)
            print('Arguments:', self.args)
            print('ix:', self.ix)
            print('iy:', self.iy)
            
    [ly,lx] = mm.shape # grab original shape of mm
    mm = np.ravel(mm)
    a = np.ravel(a)
    
    cmm = [] # chained modulus maxima array - TURN INTO PREALLOCATED STRUCTURE OF OBJECTS RATHER THAN LIST
    
    # Pre-allocate
    searched = np.zeros(mm.shape) # nonzero modulus maxima examined
    nn = np.zeros((8,1)) # nearest neighbors (8) 
    pixelArr = np.zeros((lx*ly)) # 1D vector of the pixels
    modArr = np.zeros((lx*ly)); # 1D vector of mods, maximum length is the 
    argArr = np.zeros((lx*ly)); # 1D vector of args
#     iCmm = 1 # only need this if cmm is preallocated - keeps track of how much is filled
    
    # Search for neighbors
    for i in np.arange(0,lx*ly):
        # skip empty pixels
        if mm[i] == 0:
            continue
        
        # skip already searched pixels
        if searched[i] == 1:
            continue
            
        # Mark current pixel as start of a chain
        centX = np.mod(i,lx) # calculate image x from unraveled vector
        centY = np.floor(i/lx)+1 # calculate image y from unraveled vector
        closed = 0
        pixelArr[0] = i # first pixel index
        modArr[0] = mm[i] # store first mod value
        argArr[0] = a[i] # store first arg value
        iArr = 0
        size = 1 # length counter, start at 1 for first pixel
        
#         print(centX, centY)
           
#         print('Searching for neighbors...')
        while iArr <= len(mm): # cannot be greater than length of all pixels in image
            
            # mark pixel as searched
            px = pixelArr[iArr]
            px = int(px)
            searched[px] = 1
            
            # get current image x y position
            x = np.mod(px,lx)
            y = np.floor(px/lx)+1
            
            # make sure we're not at the edges
            if x < 2 or y < 2 or x > lx-1 or y > ly-1:
                iArr = iArr+1
                break
                

            # get pixel neighbor indices
            nn[0] = int(px+lx); nn[1] = int(px+1)
            nn[2] = int(px-lx); nn[3] = int(px-1)
            nn[4] = int(nn[0]+1); nn[5] = int(nn[2]+1)
            nn[6] = int(nn[2]-1); nn[7] = int(nn[0]-1)
#             print('nn:',list(nn))
            nn = [int(a) for a in nn] # convert to integers
            
            # Find nonzero moduli neighbors that haven't been searched 
            searchednn = searched[nn] # grab searched portion
            unsearched = mm[nn]*flip_search(searchednn) # find those values that haven't been searched
            
            if np.sum(unsearched) == 0: # if nothing but zeros, there are no unsearched neighbors
                if np.sum(pixelArr[0] == nn) == 1: # if 1 of the array indices matches the first
                    closed = 1 # the chain is closed
                    break
                break # no neighbors with values
            
            # Neighbor was found otherwise:
            nnIdx = unsearched.nonzero()[0][0] # find the first neighbor Idx
#             print('Neighbor found:', nnIdx)
            
            centX = centX + x
            centY = centY + y 
            size = size+1
            iArr = iArr+1
            modArr[iArr] = mm[nn[nnIdx]]
            pixelArr[iArr] =  nn[nnIdx]
            argArr[iArr] = a[nn[nnIdx]]
        
#         print('Done searching.')
        
        # Skip closed w/ keepClosed
        if closed == 1 and keepClosed == 0: # if keepClosed is turned off and it's closed
            continue # skip it
        
        idx = iArr+1 # the ending index for slicing is iArr+1
        
        # calculate x and y from pixelArr:
        pxcoords = [int(b) for b in pixelArr[:idx]] # convert pixel coordinates to integers
        xArr = np.mod(pxcoords,lx)
        yArr = np.floor(np.divide(pxcoords,lx))+1
        
        # calculate chain parameters
        linemeanmod = np.nanmean(modArr[:idx]) # linemeanmod
        mass = linemeanmod*size # mass = lineameanmod*size
        scaledmass = mass/(2**scale) # divided by binary scale
        
        # create chain object
        newchain = chain(size, linemeanmod, mass, scaledmass, argArr[:idx], xArr[:idx], yArr[:idx]) 
        cmm.append(newchain) # store it
        
#     print('Chaining done.')
    
    # remove empty cmm entries 
    return cmm


# In[ ]:

def filter_args(args, argbuffer):
# Function that filters a numpy array of arguments for those that are pointing left (West) or right (East).
# Directions are used to refer to orientation. North would be straight up, pi/2 radians.
# INPUTS:
# - args: list of arguments
# - argbuffer: radians on either side of north/south that count as north-pointing/south-pointing (pi/3 recommended)
# OUTPUTS:
# - passedargs: the arguments that are East or West pointing
# - argfrac: the fraction of arguments that are East or West pointing
    import numpy as np
   
    north = np.pi/2; south = -1*np.pi/2
    nw = args[args > north+argbuffer] # NW pointing
    sw = args[args < south-argbuffer] # SW pointing
    east = list(set(args[args < north-argbuffer]) & set(args[args > south+argbuffer]))
    
    # calculate fraction of the args that are E- or W-pointing
    argfrac = (len(nw)+len(sw)+len(east))/len(args)
    
    nw = list(nw); sw = list(sw); east = list(east) # turn into lists for the following 
    if nw and sw and east:
        passedargs = nw+sw+east
    elif nw and sw and not east:
        passedargs = nw+sw
    elif nw and east and not sw:
        passedargs = nw+east
    elif sw and east and not nw:
        passedargs = sw+east
    elif nw and not east and not sw:
        passedargs = nw
    elif sw and not east and not nw:
        passedargs = sw
    elif east and not nw and not sw:
        passedargs = east
    else:
        passedargs = []
    
    return passedargs, argfrac




