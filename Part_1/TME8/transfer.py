import argparse
import os
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data
import torchvision
from PIL import Image
import torch.nn as nn
from sklearn.svm import LinearSVC
import torchvision.datasets as datasets
import numpy as np

torchvision.models.vgg.model_urls["vgg16"] = "http://webia.lip6.fr/~robert/cours/rdfia/vgg16-397923af.pth"
os.environ["TORCH_HOME"] = "~/datasets/"
vgg16 = torchvision.models.vgg16(pretrained=True)
PRINT_INTERVAL = 50
CUDA = False


def get_dataset(batch_size, path):
    # Cette fonction permet de recopier 3 fois une image qui
    # ne serait que sur 1 channel (donc image niveau de gris)
    # pour la "transformer" en image RGB. Utilisez la avec
    # transform.Lambda
    def duplicateChannel(img):
        img = img.convert('L')
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
        return img
    
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.NEAREST),
        duplicateChannel,
        transforms.ToTensor(),
        #transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  #mean std
    ])
    train_dataset = datasets.ImageFolder(path + '/train',
                                         transform=transform)
    val_dataset = datasets.ImageFolder(path + '/test',
                                       transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=False, pin_memory=CUDA, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                             shuffle=False, pin_memory=CUDA, num_workers=2)
    
    return train_loader, val_loader


class VGG16relu7(nn.Module):
    def __init__(self):
        super(VGG16relu7, self).__init__()
        # recopier toute la partie convolutionnelle
        self.features = nn.Sequential(*list(vgg16.features.children()))
        # garder une partie du classifieur, -2 pour s'arrêter à relu7
        self.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-2])
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def extract_features(data, model=None, path=None, size=50):
    if path is not None and os.path.isfile(path):
        print("feat load")
        X = np.load(path, allow_pickle=True)
        
        y = []
        for i, (input_, target) in enumerate(data):
            y += list(target.detach().numpy())
        
        if X.shape[0] != len(y):
            print("reduce to", X.shape[0])
            y = y[:X.shape[0]]
        
        return X, np.array(y)
    
    X = []
    y = []
    print("feat compute")
    for i, (input_, target) in enumerate(data):
        if i % PRINT_INTERVAL == 0:
            print('Batch {0:03d}/{1:03d}'.format(i, len(data)))
        if CUDA:
            input_ = input_.cuda()
        # TODO Feature extraction à faire
        X += [feat.reshape(-1).detach().numpy() for feat in model(input_)]
        y += list(target.detach().numpy())
        if i == size:
            break
        
    X = np.array(X)
    X = X / np.linalg.norm(X, axis=1).reshape((-1,1)) 
    y = np.array(y)
    
    if path is not None:
        print("feat save")
        np.save(path, X)
    
    return X, y


def main(params):
    # path = "/home/keyvan/datasets/scene15_vgg_extracted"
    path = "/home/keyvan/datasets/scene15_vgg_extracted_big"
    
    print('Instanciation de VGG16relu7')
    # TODO À remplacer par un reseau tronqué pour faire de la feature extraction
    # model = VGG16relu7()  #type: nn.Module
    model = vgg16.features  #type: nn.Module
    
    model.eval()
    if CUDA:  # si on fait du GPU, passage en CUDA
        model = model.cuda()
    
    # On récupère les données
    print('Récupération des données')
    train, test = get_dataset(params.batch_size, params.path)
    
    # Extraction des features
    print('Feature extraction')
    X_train, y_train = extract_features(train, model, path + "_train.npy")
    X_test, y_test = extract_features(test, model, path + "_test.npy")
    
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    # TODO Apprentissage et évaluation des SVM à faire
    print('Apprentissage des SVM')
    for C in np.logspace(-.5, 10., 4):
        svm = LinearSVC(C=C, max_iter=10000)
        svm.fit(X_train, y_train)
        print(svm.score(X_test, y_test))


if __name__ == '__main__':
    
    # Paramètres en ligne de commande
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='15SceneData', type=str, metavar='DIR', help='path to dataset')
    parser.add_argument('--batch-size', default=8, type=int, metavar='N', help='mini-batch size (default: 8)')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='activate GPU acceleration')
    
    args = parser.parse_args()
    if args.cuda:
        CUDA = True
        cudnn.benchmark = True
    
    main(args)
    
    input("done")
