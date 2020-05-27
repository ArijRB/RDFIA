# Chargement de la classe
from tme5 import CirclesData # import de la classe
data = CirclesData() # instancie la classe fournie
# Accès aux données
Xtrain = data.Xtrain # torch.Tensor contenant les entrées du réseau pour l'apprentissage
print(Xtrain.shape) # affiche la taille des données : torch.Size([200, 2])
N = Xtrain.shape[0] # nombre d'exemples
nx = Xtrain.shape[1] # dimensionalité d'entrée
# données disponibles : data.Xtrain, data.Ytrain, data.Xtest, data.Ytest,data.Xgrid


# Fonctions d'affichage
data.plot_data() # affiche les points de train et test
#Ygrid = forward(params, data.Xgrid) # calcul des predictions Y pour tous les points de la grille
# (forward et params non fournis, à coder)
#data.plot_data_with_grid(Ygrid) # affichage des points et de la frontière de décision gr^ace à la grille
#data.plot_loss(loss_train, loss_train, acc_train, acc_test) # affiche les courbes de loss et accuracy en train et test. 
#Les valeurs à fournir sont des scalaires,elles sont stockées pour vous,
# il suffit de passer les nouvelles valeurs à chaque itératio