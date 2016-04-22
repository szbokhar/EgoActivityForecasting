from random import shuffle
import numpy as np

def parse_file(fn, type_list):
    f = open(fn, 'r')
    data = []

    for line in f:
        svals = line.rstrip('\r\n').split(' ')
        vals = [f(a) for (f,a) in zip(type_list, svals)]
        data.append(vals)

    return data

def get_points_and_colors(fn):
    fcol = lambda x: float(x)/255
    tmp = parse_file(fn, [float, float, float, fcol, fcol, fcol, fcol])
    shuffle(tmp)
    pts = np.array(tmp)
    pts = pts[:,0:3]
    col = [l[3:6] for l in tmp]

    tmp2 = parse_file(fn+'.info', [float, float, float])
    (minx, miny, minz) = (tmp2[1][0], tmp2[1][1], tmp2[1][2])
    (maxx, maxy, maxz) = (tmp2[2][0], tmp2[2][1], tmp2[2][2])

    # Hack to crop points. Only for use with gates9
    fil = (pts[:,0]>minx)*(pts[:,1]>miny)*(pts[:,2]>minz)*(pts[:,0]<maxx)*(pts[:,1]<maxy)*(pts[:,2]<maxz)
    col = np.array(col)
    pts = pts[fil,:]
    col = col[fil,:]


    return (pts, np.array(col), np.array(tmp2))

def get_imagenames_and_path(fn):
    tmp = parse_file(fn, [lambda x: x, float, float, float])
    im = [l[0] for l in tmp]
    path = [l[1:4] for l in tmp]

    return (im, np.array(path))

def get_pathlabels(fn):
    tmp = parse_file(fn, [int])
    full = []
    [full.append(x[0]) for x in tmp]

    return np.array(full)

def get_labeldict(fn):
    print("Loading " + fn + " as dictionary")
    tmp = parse_file(fn, [int, lambda x : x])
    ldict = {}

    for r in tmp:
        ldict[r[0]] = r[1]

    return ldict
