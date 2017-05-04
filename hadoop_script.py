from pyspark import SparkContext
import HiveContext

from scipy.optimize import minimize
import numpy as np
import pandas as pd
from random import gauss
import scipy as sp
import matplotlib.pyplot as pl
%matplotlib inline

def distance(xs,ys,zs,x,y,z):

    dist = 1.0*((x - xs)**2 + (y - ys)**2 + (z - zs)**2)**0.5

    return dist

def in_picture(x,y,image_dimensions):
    # Check if point gets mapped to a pixel within the specified x and y
    # sizes of the image

    is_in_picture = (x < image_dimensions[0])*(x > 0)*(y > 0)*\
    (y < image_dimensions[1])

    return is_in_picture

def center_image_fiducials(dimensions,fiducials):

    ''' Centers image fiducial points using image dimensions  '''

    for i in range(np.shape(fiducials)[0]):
        fiducials[i,0] = fiducials[i,0] - (dimensions[0] / 2)
        fiducials[i,1] = (dimensions[1] / 2) - fiducials[i,1]

    return fiducials

def colin(params, xyz_a):

    # Unwrap params
    kappa, phi, omega, xs, ys, zs, f = params

    omega = float(omega)
    phi = float(phi) + 0.5*np.pi
    kappa = float(kappa)
    xs = float(xs)
    ys = float(ys)
    zs = float(zs)
    f = float(f)

    # -- utils
    co = np.cos(omega)
    so = np.sin(omega)
    cp = np.cos(phi)
    sp = np.sin(phi)
    ck = np.cos(kappa)
    sk = np.sin(kappa)

    a1 =  cp*ck+sp*so*sk
    b1 =  cp*sk+sp*so*ck
    c1 =  sp*co
    a2 = -co*sk
    b2 =  co*ck
    c2 =  so
    a3 =  sp*ck+cp*so*sk
    b3 =  sp*sk-cp*so*ck
    c3 =  cp*co

    ynum  = a1*(xyz_a[:,0]-xs)+b1*(xyz_a[:,1]-ys)+c1*(xyz_a[:,2]-zs)
    xnum  = a2*(xyz_a[:,0]-xs)+b2*(xyz_a[:,1]-ys)+c2*(xyz_a[:,2]-zs)
    denom = a3*(xyz_a[:,0]-xs)+b3*(xyz_a[:,1]-ys)+c3*(xyz_a[:,2]-zs)

    xx = -f*xnum/denom
    yy = f*ynum/denom

    return np.vstack([xx,yy]).T

def fullfunc(params,xyz_s,xy_t):

    ''' Find the sum of squares difference '''
    omega, phi, kappa, xs, ys, zs, f = params


    if (omega<0.0) or (omega>=2.0*np.pi):
        return 1e9+omega**4
    elif (phi<-0.5*np.pi) or (phi>=0.5*np.pi):
        return 1e9+phi**4
    elif (kappa<0.0) or (kappa>=2.0*np.pi):
        return 1e9+kappa**4
    elif zs<0.0:
        return 1e9+zs**4
    elif f<0.0:
        return 1e9+f**4
#     elif (np.abs(params[3] - 977119)>1000) or \
#             (np.abs(params[4] - 210445)>1000):
#         return 1e9 + xs**2

    colin_xy = 1.0*colin(params,xyz_s.astype(float))
    diff = ((colin_xy - xy_t)**2).sum()
    
    return diff

def call(params,xyz_s,xy_t):

    ''' Guess parameters near start and brute-force minimize '''
    start = params

    res = minimize(fullfunc, start,args=(xyz_s,xy_t), \
        method = 'Nelder-Mead', \
        options={'maxfev': 10000, 'maxiter': 10000})

    return res

guess = np.array([4.48603184, -5.75616093e-02, 0.0115, 987818.3984021898, 214563.46676424053, 800, 3000])

lidar_fiducials = np.array([
                            [988224.09, 211951.573,1494.756662], #Empire state building
                            [980598.406, 199043.071,1750.127224], #WTC
                            [987656.616, 211766.233,493.89], # 1250 Broadway
                            [983564.98, 199358.775,591.406796], # Marshall courthouse
                            [987342.468, 212511.054,380.69], #  112 West 34th St
                            [988596.086, 211789.785,255.31], # 347 5th Ave
                            [988287.232, 213228.734,488.716947]]) # 66 W 38th St

fiducials = np.array([
                        [621, 305],#Empire state building
                        [1683, 936],#WTC
                        [1185, 1400], # 1250 Broadway
                        [1217, 1143], # Marshall courthouse
                        [1860, 1637], #  112 West 34th St
                        [211, 1704], # 347 5th Ave
                        [814, 1811]])# 66 W 38th St



dimensions = np.array([1918, 2560])
fiducials = center_image_fiducials(dimensions,fiducials)

xyz_s = lidar_fiducials
xy_t = fiducials

min_score = 100000000000000
num_iter = 100
params = guess

for i in range(0, num_iter):
    result = call(params,xyz_s,xy_t)
#     print "params, score", result.x, result.fun
    if (result.fun < min_score):# and (result.x[3] < 980491):
        min_score = result.fun
        params = result.x

# print("score  = {0}\nparams = {1}".format(min_score,params))

def getStops(_, part):

#     array = np.frombuffer(bytes(part[1]))
#     reshaped = array[10:0].reshape(-1,3)
#     return reshaped.tolist()
    
    for fn,contents in part:
        a = np.fromstring(contents)[10:]
        reshaped = a.reshape(-1,3)
        yield reshaped
        break

def project(dat):  
    
   
    # Finds the desired projection
    
#     if globparams == None:
#         params = return_params(globname)
#     else:
#         params = globparams

    omega, phi, kappa, xs, ys, zs, f = params
    image_dims = dimensions
    image_dims_reversed = np.array([image_dims[1], \
        image_dims[0]])

#     # Rearrange
#     print "working on: ", filename
#     dat = np.load(filename).T.copy()

#     # Multiply by -1 because it apears as inverse; use orient?
    pixel_xy = 1.0*colin(params, dat) 

#     # un-center pixel (x,y)
    x = image_dims[0]/2 - pixel_xy[:,0].astype(int)
    y = image_dims[1]/2 + pixel_xy[:,1].astype(int)

    is_in_picture = in_picture(x,y,image_dims)

    index = np.arange(is_in_picture.size)[is_in_picture>0]

    print "npix = ", index.size

    distgrid = np.ones(image_dims_reversed)*(100000.0)
    xgrid =  -1.*np.ones(image_dims_reversed)
    ygrid = -1.*np.ones(image_dims_reversed)

    
    final_grids = [np.ones(image_dims_reversed)*(10**8), \
        -1*np.ones(image_dims_reversed), \
        -1*np.ones(image_dims_reversed)]
    
    if index.size==0:
        print "no points, returning..."
        return [distgrid, xgrid, ygrid]

    n   = distance(xs,ys,zs, dat[index,0],dat[index,1],dat[index,2]) 
    x   = x[index]
    y   = y[index]
    dat = dat[index]
    

    # Add each point to the arrays, given it is visibile (vis[i] == 1)
    # And it is closer to the camera than the current value stored in 
    # the corresponding pixel of the distance array

    nx = distgrid.shape[1]-1
    ny = distgrid.shape[0]-1

    for ii in range(index.size):
        if n[ii]<distgrid[ny-y[ii],nx-x[ii]] and n[ii]>500:
            distgrid[ny-y[ii],nx-x[ii]] = n[ii]
            xgrid[ny-y[ii],nx-x[ii]] = dat[ii,0]
            ygrid[ny-y[ii],nx-x[ii]] = dat[ii,1]

            
    out = [0, 0, 0]
#     replace = np.greater(final_grids, distgrid)
#     out[0] = final_grids[0]*np.logical_not(replace) + distgrid[0]*replace
#     out[1] = final_grids[1]*np.logical_not(replace) + distgrid[1]*replace
#     out[2] = final_grids[2]*np.logical_not(replace) + distgrid[2]*replace
#     return out      
            
#     print "Done with: ",filename
    return [distgrid, xgrid, ygrid]

image_dims = dimensions
image_dims_reversed = np.array([image_dims[1], \
        image_dims[0]])

final_grids = [np.ones(image_dims_reversed)*(10**8), \
        -1*np.ones(image_dims_reversed), \
        -1*np.ones(image_dims_reversed)]

def reducer1(final, new):
    
    out = [0, 0, 0]
    replace = np.greater(final[0], new[0])
    out[0] = final[0]*np.logical_not(replace) + new[0]*replace
    out[1] = final[1]*np.logical_not(replace) + new[1]*replace
    out[2] = final[2]*np.logical_not(replace) + new[2]*replace
    return out

sc = SparkContext()                
spark = HiveContext(sc)

rdd = sc.binaryFiles('numpy_files/').mapPartitionsWithIndex(getStops).map(project).reduce(reducer1)
