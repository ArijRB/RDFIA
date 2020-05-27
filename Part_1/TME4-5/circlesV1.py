import torch
import numpy as np
import matplotlib.pyplot as plt
from tme5 import CirclesData

def init_params(nx, nh, ny):
    params = {}
    params["Wh"] = torch.normal(0, 0.3, size=(nh, nx))
    print(params["Wh"])
    params["Wy"] = torch.normal(0,0.3,size=(ny,nh))
    params["bh"] = torch.normal(0,0.3,size=(nh,1))
    params["by"] = torch.normal(0,0.3,size=(ny,1))
    return params

def forward(params, X):
    bsize = X.size(0)
    nh = params['Wh'].size(0)
    ny = params['Wy'].size(0)
    outputs = {}
    outputs['X'] = X
    outputs['htilde'] = torch.mm(X, params['Wh'].t()) + params['bh'].t().expand(bsize, nh)
    outputs['h'] = torch.tanh(outputs['htilde'])
    outputs['ytilde'] = torch.mm(outputs['h'], params['Wy'].t()) + params['by'].t().expand(bsize, ny)
    outputs['yhat'] = torch.softmax(outputs['ytilde'],dim=0)
    return outputs['yhat'], outputs

def loss_accuracy(Yhat, Y):
    L = - torch.mean(Y * torch.log(Yhat))

    _, indYhat = torch.max(Yhat, 1)
    _, indY = torch.max(Y, 1)

    acc = torch.sum(indY == indYhat) * 100. / indY.size(0)
    return L, acc.item()

def backward(params, outputs, Y):
    bsize = Y.shape[0]
    grads = {}
    deltay = outputs['yhat'] - Y
    grads['Wy'] = torch.mm(deltay.t(), outputs['h'])
    grads['by'] = deltay.sum(0, keepdim=True).t()
    deltah = torch.mm(deltay, params['Wy']) * (1 - torch.pow(outputs['h'], 2))
    grads['Wh'] = torch.mm(deltah.t(), outputs['X'])
    grads['bh'] = deltah.sum(0, keepdim=True).t()

    grads['Wy'] /= bsize
    grads['by'] /= bsize
    grads['Wh'] /= bsize
    grads['bh'] /= bsize

    return grads

def sgd(params, grads, eta):
    params['Wy'] -= eta * grads['Wy']
    params['Wh'] -= eta * grads['Wh']
    params['by'] -= eta * grads['by']
    params['bh'] -= eta * grads['bh']

    return params



if __name__ == '__main__':

    data = CirclesData()

    data.plot_data()

    # init
    N = data.Xtrain.shape[0]
    Nbatch = 16
    nx = data.Xtrain.shape[1]
    nh = 10
    ny = data.Ytrain.shape[1]
    params = init_params(nx, nh, ny)
    # epoch
    acctests =[]
    for iteration in range(900):

        perm = np.random.permutation(N)
        Xtrain = data.Xtrain[perm, :]
        Ytrain = data.Ytrain[perm, :]

        # batches
        for j in range(N // Nbatch):
            indsBatch = range(j * Nbatch, (j+1) * Nbatch)
            X = Xtrain[indsBatch, :]
            Y = Ytrain[indsBatch, :]
            Yhat, outputs = forward(params, X)
            L, _ = loss_accuracy(Yhat, Y)
            grads = backward(params, outputs, Y)
            params = sgd(params, grads, 0.03)

        Yhat_test, _ = forward(params, data.Xtest)
        Yhat_test.detach()
        Ltest, acctest = loss_accuracy(Yhat_test, data.Ytest)    
        acctests.append(acctest)

    print ('max accuracy on test set: ' + str(max(acctests)))
