import torch
import numpy as np
import matplotlib.pyplot as plt
from tme5 import CirclesData
from torch.autograd import Variable


def init_params(nx, nh, ny):
    params = {}
    params['Wh'] = Variable(torch.randn(nh, nx) * 0.3, requires_grad=True)
    params['bh'] = Variable(torch.zeros(nh, 1), requires_grad=True)
    params['Wy'] = Variable(torch.randn(ny, nh) * 0.3, requires_grad=True)
    params['by'] = Variable(torch.zeros(ny, 1), requires_grad=True)
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



def sgd(params,eta):
    with torch.no_grad(): # Attention a bien utiliser torch.no_grad()
        params['Wy'] -= eta * params['Wy'].grad
        params['Wy'].grad.zero_()
        params['Wh'] -= eta * params['Wh'].grad
        params['Wh'].grad.zero_()
        params['by'] -= eta *  params['by'].grad
        params['by'].grad.zero_()
        params['bh'] -= eta * params['bh'].grad
        params['bh'].grad.zero_()

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
    for iteration in range(100):

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
            L.backward()
            params = sgd(params,0.03)

        Yhat_test, _ = forward(params, data.Xtest)
        Yhat_test.detach()
        Ltest, acctest = loss_accuracy(Yhat_test, data.Ytest)
        acctests.append(acctest)

    print ('max accuracy on test set: ' + str(max(acctests)))
   