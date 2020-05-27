from TME1.tools2 import *
from sift import *
from kmeans import *

dir_sc = os.path.join('data', 'Scene')
dir_sift = os.path.join('data', 'sift')
path_vdict = os.path.join('data', 'kmeans', 'vdict.npy')
path_vdsift = os.path.join('data', 'kmeans', 'vdsift.npy')
path_vdinames = os.path.join('data', 'kmeans', 'vdinames.npy')

inames, ilabls, cnames = load_dataset(dir_sc)

vdict = compute_load_vdict(dir_sc, dir_sift, inames, compute_sift_image, path_vdict, compute_vdict)
print("Done")
