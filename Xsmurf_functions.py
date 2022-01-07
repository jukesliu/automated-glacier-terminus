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
    
    count = 0 # FOR TESTING ONlY
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
            
        # check if the nearest modulus value works
        if (np.abs(dx_norm[i]) > err_deriv) or (np.abs(dy_norm[i]) > err_deriv):
            xul = np.mod(i,lx) + dx_norm[i] + 0.5
            yul = np.floor(i/lx) + dy_norm[i] + 0.5
                       
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
        
        if m[i] <= m_interp:
            continue
        
        # Second point interpolated (notice sign switch to subtracting. dx_norm)
        x_interp = np.mod(i,lx) - dx_norm[i]
        y_interp = np.floor(i/lx) - dy_norm[i]
#         print(x_interp); count = count + 1
        
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

def calculate_shifts(x_interp, y_interp):
    # calculate shifts for bilinear interpolation
    # SYNTAX: [sx, sy, sxsy] = calculate_shifts(x_interp, y_interp)
    import numpy as np
    sx = x_interp - np.fix(x_interp)
    sy = y_interp - np.fix(y_interp)    
    sxsy = sx*sy
    return sx, sy, sxsy

    
def calculate_coeff(sx, sy, sxsy):
    # calculate bilinear interpolation coefficients given outputs from calculate_shifts()
    # SYNTAX: [c00, c10, c01, c11] = calculate_coeff(sx, sy, sxsy)
    import numpy as np
    c00 = 1-sx-sy+sxsy
    c10 = sx-sxsy
    c01 = sy-sxsy
    c11 = sxsy
    
    return c00, c10, c01, c11

def ordered_set(nparray):
    # Retain index order with set function
    import numpy as np
    os = sorted(set(list(nparray)), key=list(nparray).index)
    
    return np.array(os)

def wtmm2d_v(I,wavelet,scale):
    import numpy as np
    from ttictoc import tic,toc
    
    # Perform 2D Wavelet Transform Modulus Maxima (WTMM) Method on image I
    # with wavelet at a user-specified scale. This maxima interpolation is vectorized
    # in this version of the function, which means it runs ~1.5x faster.
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
    # Last Modified: 2021 10 22
    
#     tic()
    I = pad_square(np.array(I))
    
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
    
    ########################################################################################################
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
    mm = np.zeros(mshape) # initialize mm
    idxs = np.arange(0, lx*ly) # vector of idxs for m
    idxs = recon(mshape, idxs)
    
    for i in range(0, lx):
        # select the columns
        mcol = m[:,i] 
        idxcol = idxs[:,i]
        dx_norm_col = dx_norm[:,i]
        dy_norm_col = dy_norm[:,i]
    
        # unravel m
        mr = np.ravel(m, order='F')
        
         # stay away from image edges
        if (np.floor(idxcol/lx) == 0).all() or (np.floor(idxcol/lx) == ly-1).all():
    #         print('Vertical edge found at '+str(i))
            continue # skip vertical edges
        mcol2 = mcol[1:-1]; idxcol2 = idxcol[1:-1] # remove the horizontal edge
        dx_norm2 = dx_norm_col[1:-1]; dy_norm2 = dy_norm_col[1:-1]
         
        # remove small moduli (keep only m > err_norm)
        mcol3 = mcol2[mcol2 >= err_norm]; idxcol3 = idxcol2[mcol2 >= err_norm]
        dx_norm3 = dx_norm2[mcol2 >= err_norm]; dy_norm3 = dy_norm2[mcol2 >= err_norm]

        # check if the nearest modulus value works
        # where dx_norm > err_deriv concatenated with where dy_norm > err_deriv 
        dx_norm4 = np.concatenate((dx_norm3[(np.abs(dx_norm3) > err_deriv)], 
                                  dx_norm3[(np.abs(dy_norm3) > err_deriv)])); dx_norm4 = ordered_set(dx_norm4) 
        dy_norm4 = np.concatenate((dy_norm3[(np.abs(dx_norm3) > err_deriv)],
                                  dy_norm3[(np.abs(dy_norm3) > err_deriv)])); dy_norm4 = ordered_set(dy_norm4)   
        mcol4 = np.concatenate((mcol3[(np.abs(dx_norm3) > err_deriv)],
                               mcol3[(np.abs(dy_norm3) > err_deriv)])); mcol4 = ordered_set(mcol4)
        idxcol4 = np.concatenate((idxcol3[(np.abs(dx_norm3) > err_deriv)],
                                 idxcol3[(np.abs(dy_norm3) > err_deriv)])); idxcol4 = ordered_set(idxcol4)
        
        # add these to the chain:
        xul = np.mod(idxcol4,lx) + dx_norm4 + 0.5
        yul = np.floor(idxcol4/lx) + dy_norm4 + 0.5
        
        mcol5 = mcol4[mcol4 > mr[(np.fix(xul).astype(int)+np.fix(yul)*lx).astype(int)]]
        idxcol5 = idxcol4[mcol4 > mr[(np.fix(xul).astype(int)+np.fix(yul)*lx).astype(int)]]
        dx_norm5 = dx_norm4[mcol4 > mr[(np.fix(xul).astype(int)+np.fix(yul)*lx).astype(int)]]
        dy_norm5 = dy_norm4[mcol4 > mr[(np.fix(xul).astype(int)+np.fix(yul)*lx).astype(int)]]

        # note sign difference in operation
        xul = np.mod(idxcol5,lx) - dx_norm5 + 0.5
        yul = np.floor(idxcol5/lx) - dy_norm5 + 0.5
        
        # last pass
        mcol6 = mcol5[mcol5 >= mr[(np.fix(xul).astype(int)+np.fix(yul)*lx).astype(int)]]
        idxcol6 = idxcol5[mcol5 >= mr[(np.fix(xul).astype(int)+np.fix(yul)*lx).astype(int)]]
        dx_norm6 = dx_norm5[mcol5 >= mr[(np.fix(xul).astype(int)+np.fix(yul)*lx).astype(int)]]
        dy_norm6 = dy_norm5[mcol5 >= mr[(np.fix(xul).astype(int)+np.fix(yul)*lx).astype(int)]]    
    
        if mcol6.size > 0: # any moduli remain, they are modulus maxima: 
            xs = np.mod(idxcol6,lx)
            ys = np.floor(idxcol6/lx)
            mm[xs.astype(int),ys.astype(int)] = mcol6
    
        # For all remaining m, proceed with the bilinear inteprolation
        # rename vector variable names to m1, idx1, dynorm1, dxnorm1 for simplicity
        m1 = np.array([b for b in mcol4 if b not in mcol6])
        m1 = np.array([b for b in mcol3 if b not in m1])      
        idx1 = np.array([c for c in idxcol4 if c not in idxcol6])
        idx1 = np.array([c for c in idxcol3 if c not in idx1])             
        dxnorm1 = np.array([d for d in dx_norm4 if d not in dx_norm6])
        dxnorm1 = np.array([d for d in dx_norm3 if d not in dxnorm1])
        dynorm1 = np.array([e for e in dy_norm4 if e not in dy_norm6])
        dynorm1 = np.array([e for e in dy_norm3 if e not in dynorm1])
        
        x_interp = np.mod(idx1,lx) + dxnorm1
        y_interp = np.floor(idx1/lx) + dynorm1

        # stay away from edges during interpolation
        m2 = np.concatenate((m1[x_interp >= 0], m1[x_interp < lx-1],
                             m1[y_interp >= 0], m1[y_interp < ly-1])); m2 = ordered_set(m2)
        idx2 = np.concatenate((idx1[x_interp >= 0], idx1[x_interp < lx-1],
                               idx1[y_interp >= 0], idx1[y_interp < ly-1])); idx2 = ordered_set(idx2)                          
        dxnorm2 = np.concatenate((dxnorm1[x_interp >= 0], dxnorm1[x_interp < lx-1],
                                  dxnorm1[y_interp >= 0], dxnorm1[y_interp < ly-1])); dxnorm2 = ordered_set(dxnorm2)                     
        dynorm2 = np.concatenate((dynorm1[x_interp >= 0], dynorm1[x_interp < lx-1],
                                  dynorm1[y_interp >= 0], dynorm1[y_interp < ly-1])); dynorm2 = ordered_set(dynorm2)
        
        [sx, sy, sxsy] = calculate_shifts(x_interp, y_interp) # calculate shifts
        [c00, c10, c01, c11] = calculate_coeff(sx, sy, sxsy) # calculate bilinear interpolation coefficients
        
        # We compare the modulus of the point with the
        # interpolated modulus. It must be larger to be
        # still considered as a potential gradient extrema.

        # Here, we consider that it is strictly superior.
        # The next comparison will be superior or equal.
        # This way, the extrema is in the light part of the image.
        # By inverting both tests, we can put it in the
        # dark side of the image.
        ul = (np.fix(x_interp) + np.fix(y_interp)*lx).astype(int)
        m_interp = mr[ul]*c00 + mr[ul+1]*c10 + mr[ul+lx]*c01 + mr[ul+lx+1]*c11 # +1 removed from indexing
        
        # Keep only those where m > m_interp:
        m3 = m2[m2 > m_interp]; idx3 = idx2[m2 > m_interp]
        dxnorm3 = dxnorm2[m2 > m_interp]; dynorm3 = dynorm2[m2 > m_interp]
        sx2 = sx[m2 > m_interp]; sy2 = sy[m2 > m_interp]; sxsy2 = sxsy[m2 > m_interp]
            
        # Second point interpolated (notice sign switch to subtracting dx_norm), superior or equal comparison
        x_interp2 = np.mod(idx3,lx) - dxnorm3 
        y_interp2 = np.floor(idx3/lx) - dynorm3        

        # stay away from image edges
        m4 = np.concatenate((m3[x_interp2 >= 0], m3[x_interp2 < lx-1],
                                 m3[y_interp2 >= 0], m3[y_interp2 < ly-1])); m4 = ordered_set(m4)
        idx4 = np.concatenate((idx3[x_interp2 >= 0], idx3[x_interp2 < lx-1],
                                   idx3[y_interp2 >= 0], idx3[y_interp2 < ly-1])); idx4 = ordered_set(idx4) 
        dxnorm4 = np.concatenate((dxnorm3[x_interp2 >= 0], dxnorm3[x_interp2 < lx-1],
                                      dxnorm3[y_interp2 >= 0], dxnorm3[y_interp2 < ly-1])); dxnorm4 = ordered_set(dxnorm4)
        dynorm4 = np.concatenate((dynorm3[x_interp2 >= 0], dynorm3[x_interp2 < lx-1],
                                      dynorm3[y_interp2 >= 0], dynorm3[y_interp2 < ly-1])); dynorm4 = ordered_set(dynorm4)
        sx3 = np.concatenate((sx2[x_interp2 >= 0], sx2[x_interp2 < lx-1],
                                 sx2[y_interp2 >= 0], sx2[y_interp2 < ly-1])); sx3 = ordered_set(sx3)
        sy3 = np.concatenate((sy2[x_interp2 >= 0], sy2[x_interp2 < lx-1],
                                 sy2[y_interp2 >= 0], sy2[y_interp2 < ly-1])); sy3 = ordered_set(sy3)
        sxsy3 = sx3*sy3
        [c00, c10, c01, c11] = calculate_coeff(sx3, sy3, sxsy3)
        
        ul2 = (np.fix(x_interp2) + np.fix(y_interp2)*lx).astype(int) 
        m_interp2 = mr[ul2]*c11 + mr[ul2+1]*c01 + mr[ul2+lx]*c10 + mr[ul2+lx+1]*c00 # different coefficients
        
        # Keep only those where m >= m_interp:
        m5 = m4[m4 >= m_interp2]; idx5 = idx4[m4 >= m_interp2]

        if m5.size > 0: # any moduli remain, they are modulus maxima: 
            xs = np.mod(idx5,lx); ys = np.floor(idx5/lx)
            mm[xs.astype(int),ys.astype(int)] = m5

#     print(toc())
    return dx, dy, mm, m, a


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


def wtmmchains(mm, a, keepClosed, scale, counter):
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
        def __init__(self, size, linemeanmod, mass, scaledmass, args, ix, iy, scale):
            self.size = size
            self.linemeanmod = linemeanmod
            self.mass = mass
            self.scaledmass = scaledmass
            self.args = args
            self.ix = ix
            self.iy = iy
            self.scale = scale

        def show(self):
            print('Size:', self.size)
            print('Linemeanmod:', self.linemeanmod)
            print('Mass:', self.mass)
            print('Metric:', self.metric)
            print('Arguments:', self.args)
            print('ix:', self.ix)
            print('iy:', self.iy)
            print('scale number:', scale)
            
    [ly,lx] = mm.shape # grab original shape of mm
    mm = np.ravel(mm)
    a = np.ravel(a)
    
    cmm = [] # chained modulus maxima array
    
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
            if x < 2 or y < 2 or x > lx-2 or y > ly-2:
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
        newchain = chain(size, linemeanmod, mass, scaledmass, argArr[:idx], xArr[:idx], yArr[:idx],counter) 
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


def diff_idx(array1, array2):
    # records the indexes of array1 for elements that are not in array 2
    import numpy as np
    
    idxs = []
    for i in range(0, len(array1)):
        val = array1[i]
        if val not in array2:
            idxs.append(i)
    return idxs


def wtmm2d_v2(I,wavelet,scale):
    import numpy as np
    from ttictoc import tic,toc
    
    # Perform 2D Wavelet Transform Modulus Maxima (WTMM) Method on image I
    # with wavelet at a user-specified scale. This maxima interpolation is vectorized
    # in this version of the function, which means it runs ~1.5x faster.
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
    # Last Modified: 2021 10 22
    
#     tic()
    I = pad_square(np.array(I))
    
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
    
    ########################################################################################################
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
    mm = np.zeros(mshape) # initialize mm
    idxs = np.arange(0, lx*ly) # vector of idxs for m
    idxs = recon(mshape, idxs)
    
    for i in range(0, lx):
        # select the columns
        mcol = m[:,i] 
        idxcol = idxs[:,i]
        dx_norm_col = dx_norm[:,i]
        dy_norm_col = dy_norm[:,i]
    
        # unravel m
        mr = np.ravel(m, order='F')
        
         # stay away from image edges
        if (np.floor(idxcol/lx) == 0).all() or (np.floor(idxcol/lx) == ly-1).all():
    #         print('Vertical edge found at '+str(i))
            continue # skip vertical edges
        mcol2 = mcol[1:-1]; idxcol2 = idxcol[1:-1] # remove the horizontal edge
        dx_norm2 = dx_norm_col[1:-1]; dy_norm2 = dy_norm_col[1:-1]
         
        # remove small moduli (keep only m > err_norm)
        mask2 = np.where(mcol2 >= err_norm)
        # keep:
        mcol3 = mcol2[mask2]; idxcol3 = idxcol2[mask2]
        dx_norm3 = dx_norm2[mask2]; dy_norm3 = dy_norm2[mask2]

        # check if the nearest modulus value works
        # where dx_norm > err_deriv and dy_norm > err_deriv 
        mask3 = np.where((np.abs(dx_norm3) > err_deriv) & (np.abs(dy_norm3) > err_deriv))
        # keep:
        dx_norm4 = dx_norm3[mask3]; dy_norm4 = dy_norm3[mask3]
        mcol4 = mcol3[mask3]; idxcol4 = idxcol3[mask3]

        # add these to the chain:
        xul = np.mod(idxcol4,lx) + dx_norm4 + 0.5
        yul = np.floor(idxcol4/lx) + dy_norm4 + 0.5
        
        mask4 = np.where(mcol4 > mr[(np.fix(xul).astype(int)+np.fix(yul)*lx).astype(int)])
        mcol5 = mcol4[mask4]; idxcol5 = idxcol4[mask4]
        dx_norm5 = dx_norm4[mask4]; dy_norm5 = dy_norm4[mask4]

        # note sign difference in operation
        xul = np.mod(idxcol5,lx) - dx_norm5 + 0.5
        yul = np.floor(idxcol5/lx) - dy_norm5 + 0.5
        
        # last pass
        mask5 = np.where(mcol5 >= mr[(np.fix(xul).astype(int)+np.fix(yul)*lx).astype(int)])
        # keep:
        mcol6 = mcol5[mask5]; idxcol6 = idxcol5[mask5]
        dx_norm6 = dx_norm5[mask5]; dy_norm6 = dy_norm5[mask5]  
    
        if mcol6.size > 0: # any moduli remain, they are modulus maxima: 
            xs = np.mod(idxcol6,lx)
            ys = np.floor(idxcol6/lx)
            mm[xs.astype(int),ys.astype(int)] = mcol6
    
        # For all remaining m, proceed with the bilinear inteprolation
        # Remaining m:
        diffidx1 = diff_idx(mcol4, mcol6)
        diffidx2 = diff_idx(mcol3, mcol4[diffidx1])
        # rename vector variable names to m1, idx1, dynorm1, dxnorm1 for simplicity
        m1 = mcol3[diffidx2]
        idx1 = idxcol3[diffidx2]
        dxnorm1 = dx_norm3[diffidx2]
        dynorm1 = dy_norm3[diffidx2]
        
        x_interp = np.mod(idx1,lx) + dxnorm1
        y_interp = np.floor(idx1/lx) + dynorm1

        # stay away from edges during interpolation
        mask6 = np.where((x_interp >= 0) & (x_interp < lx-1) & (y_interp >= 0) & (y_interp < ly-1))
        # Keep:
        m2 = m1[mask6]; idx2 = idx1[mask6]
        dxnorm2 = dxnorm1[mask6]; dynorm2 = dynorm1[mask6]
        x_interp = x_interp[mask6]; y_interp = y_interp[mask6] # make sure dimensions align
        
        [sx, sy, sxsy] = calculate_shifts(x_interp, y_interp) # calculate shifts
        [c00, c10, c01, c11] = calculate_coeff(sx, sy, sxsy) # calculate bilinear interpolation coefficients
        
        # We compare the modulus of the point with the
        # interpolated modulus. It must be larger to be
        # still considered as a potential gradient extrema.

        # Here, we consider that it is strictly superior.
        # The next comparison will be superior or equal.
        # This way, the extrema is in the light part of the image.
        # By inverting both tests, we can put it in the
        # dark side of the image.
        
        ul = (np.fix(x_interp) + np.fix(y_interp)*lx).astype(int)
        m_interp = mr[ul]*c00 + mr[ul+1]*c10 + mr[ul+lx]*c01 + mr[ul+lx+1]*c11 # +1 removed from indexing
        
        # Keep only those where m > m_interp:
        mask7 = np.where(m2 > m_interp)
        # Keep:
        m3 = m2[mask7]; idx3 = idx2[mask7]
        dxnorm3 = dxnorm2[mask7]; dynorm3 = dynorm2[mask7]
        sx2 = sx[mask7]; sy2 = sy[mask7]; sxsy2 = sxsy[mask7]

        # Second point interpolated (notice sign switch to subtracting dx_norm), superior or equal comparison
        x_interp2 = np.mod(idx3,lx) - dxnorm3 
        y_interp2 = np.floor(idx3/lx) - dynorm3        

        # stay away from image edges
        mask8 = np.where((x_interp2 >= 0) & (x_interp2 < lx-1) & (y_interp2 >= 0) & (y_interp2 < ly-1))
        # Keep:
        m4 = m3[mask8]; idx4 = idx3[mask8]
        dxnorm4 = dxnorm3[mask8]; dynorm4 = dynorm3[mask8]
        sx3 = sx2[mask8]; sy3 = sy2[mask8];
        x_interp2 = x_interp2[mask8]; y_interp2 = y_interp2[mask8] # make sure dimensions align
        
        sxsy3 = sx3*sy3
        [c00, c10, c01, c11] = calculate_coeff(sx3, sy3, sxsy3)
        
        ul2 = (np.fix(x_interp2) + np.fix(y_interp2)*lx).astype(int)
        m_interp2 = mr[ul2]*c11 + mr[ul2+1]*c01 + mr[ul2+lx]*c10 + mr[ul2+lx+1]*c00 # different coefficients
        
        # Keep only those where m >= m_interp:
        m5 = m4[m4 >= m_interp2]; idx5 = idx4[m4 >= m_interp2]

        if m5.size > 0: # any moduli remain, they are modulus maxima: 
            xs = np.mod(idx5,lx); ys = np.floor(idx5/lx)
            mm[xs.astype(int),ys.astype(int)] = m5

#     print(toc())
    return dx, dy, mm, m, a


def wtmm2d_img(image):
    if True == True:
        topchains_dfs = []
        
        img = Image.open(processed_image_path+image)
#         print(str(image_num)+' out of '+str(len(imagelist))+' '+image)
        print(image)
        
        # WTMM
        counter = 0
        all_cmm = [] # to hold all the chains produced
        # ascend over all scales
        for iOct in np.arange(0, nOct):
            for iVox in np.arange(0, nVox):

                # calculate scale in pixels
                scale = 6/0.86*amin*2**(iOct+(iVox/nVox))
#                 print('Scale: '+str(scale))

                # wavelet transform
                [dx, dy, mm, m, a] = wtmm2d_v2(img, wavelet, scale)
                
                # emask
                masked_a = emask(box_array, a)
                masked_mm = emask(box_array, mm)
                masked_m = emask(box_array, m)

                # chain
                cmm = wtmmchains(masked_mm,masked_a,0,scale,counter)

                # increment
                all_cmm.extend(cmm)
                counter = counter +1 
       
    
        # Make directory to store chain jsons:
        imgfolder = processed_image_path+image+'_chains/'
        if not os.path.exists(imgfolder):
            os.mkdir(imgfolder)

        # Pick the terminus line
        # Find maximum mods and sizes for thresholding
        mods = []; sizes = []
        for chain in all_cmm:
            sizes.append(chain.size)
            mods.append(chain.linemeanmod)
        maxmod = np.nanmax(mods); maxsize = np.nanmax(sizes)
            
        mass_or_size = []
        passed_chains = []
        passcount = 0
        for chain in all_cmm:
            if chain.linemeanmod > mod_thresh*maxmod: # only chains that pass the mod threshold
#                 if chain.size > size_thresh*maxsize: # only chains that pass the size threshold
                if chain.size > size_thresh*np.sqrt(len(box_array[box_array > 0])):
                    [passedargs, argfrac] = filter_args(chain.args, np.pi/3) # identify the left & right-pointing args
                    if argfrac > arg_thresh: # only chains that pass the orientation threshold
                        if metric == 0:
                            mass_or_size.append(chain.mass)
                        elif metric == 1:
                            mass_or_size.append(chain.scaledmass)
                        else:
                            mass_or_size.append(chain.size)
                        passcount += 1
                        passed_chains.append(chain)
        
        if passcount > 0: # if chains remain:
            # sort by mass or size and grab the top 5
            zipped = zip(mass_or_size, passed_chains)              
            top_chains = sorted(zipped,reverse=True,
                                key=lambda zipped: zipped[0])[:5] # sort chains that passed

            # grab info from top 5 chains
            scales = []; boxids = []; orders = []; scenes = []; dates = []
            # write the top 5 to json
            for chain in top_chains:
                # grab the chain
                chain = chain[1]

                # convert dtypes to json serializable dtypes:
                chain.size = int(chain.size)
                chain.linemeanmod = float(chain.linemeanmod)
                chain.mass = float(chain.mass)
                chain.scaledmass = float(chain.scaledmass)
                chain.args = list(map(float, chain.args))
                chain.ix = list(map(int, chain.ix))
                chain.iy = list(map(int, chain.iy))
                chain.scale = str(chain.scale)
                scales.append(chain.scale.zfill(3))

                # write object to json file
                with open(imgfolder+chain.scale.zfill(3)+'_chain.json', 'w') as f:
                    json.dump(chain.__dict__, f)

            topchains_df = pd.DataFrame(top_chains,columns=['Metric','chain'])
            rows = len(topchains_df)

            for n in range(0,rows):
                boxids.append(BoxID.zfill(3)) # box string
                order = n+1 # order of chains (already sorted)
                orders.append(order)
                scenes.append(image[2:-20])
                date = datetime.datetime.strptime(image[19:27], '%Y%m%d')
                date = date.strftime("%Y-%m-%d"); dates.append(date)
            topchains_df['BoxID'] = boxids; topchains_df['Scene'] = scenes
            topchains_df['datetimes'] = dates;
            topchains_df['Scale'] = scales; topchains_df['Order'] = orders
            topchains_df = topchains_df[['BoxID','Scene','datetimes','Scale','Metric','Order']]
            topchains_dfs.append(topchains_df)

            # visualize top chains:
            colors = pl.cm.viridis(np.linspace(0,1,5)) # generate colors using a colormap
            plt.figure(figsize=(8,8))
            plt.imshow(np.array(img), aspect='equal', cmap = 'gray')
            plt.gca().set_aspect('equal'); plt.gca().invert_yaxis()
            for k in range(0, len(top_chains)): # plot chains (purple = top, yellow = 5th)
                plt.plot(top_chains[len(top_chains)-1-k][1].ix, 
                         top_chains[len(top_chains)-1-k][1].iy, 's-', color=colors[k],markersize=0.1)
            plt.show()
            return topchains_dfs
        else:
            print('No chains passed.')


