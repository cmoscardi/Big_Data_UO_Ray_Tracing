from pyspark import SparkContext
import numpy as np

def distance(xs,ys,zs,x,y,z):

    dist = 1.0*((x - xs)**2 + (y - ys)**2 + (z - zs)**2)**0.5
    return dist

def in_picture(x,y,image_dimensions):
    # Check if point gets mapped to a pixel within the specified x and y
    # sizes of the image

    is_in_picture = (x < image_dimensions[0])*(x > 0)*(y > 0)*\
    (y < image_dimensions[1])

    return is_in_picture

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

def project(dat):


    # Finds the desired projection

    params = [1.345794960057916, 0.056072502823257861, -0.012072660480989369, 988504.86108153069,
                  214494.40203705645, 799.41113612974959, 2831.8189679364423]

    omega, phi, kappa, xs, ys, zs, f = params
    image_dims = [1918, 2560]
    image_dims_reversed = np.array([image_dims[1], image_dims[0]])

    # Multiply by -1 because it apears as inverse; use orient?
    pixel_xy = 1.0*colin(params, dat)

    # un-center pixel (x,y)
    x = image_dims[0]/2 + pixel_xy[:,0].astype(int)
    y = image_dims[1]/2 + pixel_xy[:,1].astype(int)

    is_in_picture = in_picture(x,y,image_dims)

    index = np.arange(is_in_picture.size)[is_in_picture>0]

    distgrid = np.ones(image_dims_reversed)*(100000.0)
    xgrid =  -1.*np.ones(image_dims_reversed)
    ygrid = -1.*np.ones(image_dims_reversed)


    if index.size==0:
        return [distgrid, xgrid, ygrid]

    n   = distance(xs,ys,zs, dat[index,0],dat[index,1],dat[index,2])
    x   = x[index]
    y   = y[index]
    dat = dat[index]


    # Add each point to the arrays, given it is visibile (vis[i] == 1)
    # And it is closer to the camera than the current value stored in 
    # the corresponding pixel of the distance array


    for ii in range(index.size):
        if n[ii]<distgrid[y[ii], x[ii]] and n[ii]>500:
                
            distgrid[y[ii],x[ii]] = n[ii]
            xgrid[y[ii],x[ii]] = dat[ii,0]
            ygrid[y[ii],x[ii]] = dat[ii,1]

    return [distgrid, xgrid, ygrid]

def mapper1(part):
                    
    for fn, contents in part:
        reshaped = np.fromstring(contents)[10:]
        reshaped = reshaped.reshape(-1,3)
        vals = project(reshaped)
        yield  vals


def  reducer1(final,new):

    out = [0, 0, 0]
    replace = np.greater(final[0], new[0])
    out[0] = final[0]*np.logical_not(replace) + new[0]*replace
    out[1] = final[1]*np.logical_not(replace) + new[1]*replace
    out[2] = final[2]*np.logical_not(replace) + new[2]*replace
    return out

if __name__ == "__main__":
    
    sc = SparkContext()
    grids = sc.binaryFiles('project/data_npy78/',99).mapPartitions(mapper1).treeReduce(reducer1)
                
    np.save('distgrid_new.npy', grids[0])
    np.save('xgrid_new.npy', grids[1])
    np.save('ygrid_new.npy', grids[2])
    print "Done"

