from __future__ import print_function
import sys

def main(path_to_nvm):

    nvm = open(path_to_nvm, 'r')
    camfile = open('cameras.txt', 'w')
    pointfile = open('points.txt', 'w')

    nvm.readline()
    nvm.readline()

    numcams = int(nvm.readline())
    print(("number of cameras", numcams))

    for i in range(numcams):
        line = nvm.readline()
        line = line.replace('\t', ' ')
        parts = line.split(' ')
        outline = ' '.join([parts[0], parts[6], parts[7], parts[8]])
        print(outline, file=camfile)


    nvm.readline()

    numpoints = int(nvm.readline())
    print(("number of points", numpoints))

    for i in range(numpoints):
        line = nvm.readline()
        line = line.replace('\t', ' ')
        parts = line.split(' ')
        print(' '.join(parts[0:6]), file=pointfile)

    nvm.close()
    camfile.close()
    pointfile.close()


if __name__ == "__main__":
    path_to_nvm = sys.argv[1]
    print(("Loading nvm file", path_to_nvm))
    main(path_to_nvm)
