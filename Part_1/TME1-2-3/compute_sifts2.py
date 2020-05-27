from TME1.tools2 import *
"""
dir_sc = os.path.join('data', 'Scene')
dir_sift = os.path.join('data', 'sift')
path_vdict = os.path.join('data', 'kmeans', 'vdict.npy')
path_vdsift = os.path.join('data', 'kmeans', 'vdsift.npy')
path_vdinames = os.path.join('data', 'kmeans', 'vdinames.npy')

inames, ilabls, cnames = load_dataset(dir_sc)

sifts_list_by_image = compute_load_sift_dataset(dir_sc, dir_sift, inames, compute_sift_image)
"""
def compute_grad(I):
    #noyau_x = np.array([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]]) * 1/4
    #noyau_y = np.array([[-1.0, -2, -1], [0, 0, 0], [1, 2, 1]]) * 1/4
    noyau_x_hx = np.array([-1, 0, 1]).reshape((1, 3))
    noyau_x_hy = np.array([1, 2, 1]).reshape((3, 1))
    
    noyau_y_hx = np.array([1, 2, 1]).reshape((1, 3))
    noyau_y_hy = np.array([-1, 0, 1]).reshape((3, 1))
    
    #print(np.matmul(noyau_x_hy,noyau_x_hx))
    
    Ix = conv_separable(I, noyau_x_hx, noyau_x_hy)
    Iy = conv_separable(I, noyau_y_hx, noyau_y_hy)
    return Ix, Iy

def compute_grad_mod_ori(I):
    g_x, g_y = compute_grad(I)
    grad_norme = np.sqrt(np.multiply(g_x, g_x) + np.multiply(g_y, g_y))
    grad_orientation_disc = compute_grad_ori(g_x, g_y, grad_norme)
    return grad_norme, grad_orientation_disc

def compute_sift_region(grad_norm, grad_ori_disc, mask=None):
    """retourne l'encodage de la région sous la forme d'un vecteur de taille 128
    mask: """
    l = 16
    n_orientation = 8
    assert grad_norm.shape == (l,l), grad_norm.shape
    assert grad_ori_disc.shape == (l,l), grad_ori_disc.shape
    assert np.all((0<=grad_ori_disc) & (grad_ori_disc<n_orientation) | (grad_ori_disc==-1))
        
    if mask is not None:
        grad_norm = grad_norm.multiply(mask)
    
    # 4x4 histogrammes de 8 cases
    taille_case = 4
    histo = np.zeros((l//taille_case, l//taille_case, n_orientation))
    for i in range(grad_norm.shape[0]):
        for j in range(grad_norm.shape[1]):
            if grad_ori_disc[i][j] != -1:
                histo[i//taille_case][j//taille_case][grad_ori_disc[i][j]] += grad_norm[i][j]
    
    # postprocessing
    encodage = histo.reshape((128,))
    
    # retourne le vecteur nul si la norme du vecteur est inférieure à .5
    norm = np.linalg.norm(encodage)
    if norm < .5:
        return np.zeros(128)
    
    # normalise
    encodage /= norm
    
    # seuillage
    seuil = .2
    encodage = np.minimum(encodage, seuil)
    
    encodage /= np.linalg.norm(encodage)
    
    return encodage

def compute_sift_image(I):
    """pas scale invariant"""
    x, y = dense_sampling(I)
    im = auto_padding(I)
    grad_norm, grad_ori_disc = compute_grad_mod_ori(im)
    sifts = np.empty((len(x), len(y), 128))
    shift_x = shift_y = 16
    
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            # SIFT du patch de coordonnées (xi, yj)
            gn = grad_norm    [xi: xi + shift_x, yj: yj + shift_y]
            go = grad_ori_disc[xi: xi + shift_x, yj: yj + shift_y]
            sifts[i, j, :] = compute_sift_region(gn, go, mask=None)
    return sifts

def _test_compute_sift_image():
    from TME1.tools2 import load_dataset
    dir_sc = os.path.join('data', 'Scene')
    dir_sift = os.path.join('data', 'sift')
    inames, ilabls, cnames = load_dataset(dir_sc)
    print(len(inames))
    exit()
    compute_load_sift_dataset(dir_sc, dir_sift, inames, compute_sift_image)

if __name__ == '__main__':
    _test_compute_sift_image()