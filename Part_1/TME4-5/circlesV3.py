import torch
import numpy as np
import matplotlib.pyplot as plt
from tme5 import CirclesData
from torch.autograd import Variable


def init_model(nx, nh, ny):
    model = torch.nn.Sequential(
    torch.nn.Linear(nx, nh),
    torch.nn.Tanh(),
    torch.nn.Linear(nh, ny),
    )
    loss = torch.nn.CrossEntropyLoss()
    return model, loss

def loss_accuracy(Yhat, Y,loss):
    L = loss(Yhat, Y)
    acc = torch.mean((Yhat.argmax(1) == Y).float())
    return L, acc



def sgd(model,eta):
    with torch.no_grad(): # Attention a bien utiliser torch.no_grad()
        for param in model.parameters():
            param.data -= eta * param.grad.data
        model.zero_grad()



if __name__ == '__main__':

    data = CirclesData()

    data.plot_data()

    # init
    N = data.Xtrain.shape[0]
    Nbatch = 16
    nx = data.Xtrain.shape[1]
    nh = 10
    ny = data.Ytrain.shape[1]
    model, loss = init_model(nx, nh, ny)
  

    # epoch
    acctests =[]
    for iteration in range(100):

        perm = np.random.permutation(N)
        Xtrain = data.Xtrain[perm, :]
        Ytrain = data.Ytrain[perm, :]
        Ytrain = Ytrain.argmax(1)
        # batches
        for j in range(N // Nbatch):
            indsBatch = range(j * Nbatch, (j+1) * Nbatch)
            X = Xtrain[indsBatch, :]
            Y = Ytrain[indsBatch]
            Yhat= model(X)
            L, _ = loss_accuracy(Yhat,Y,loss)
            L.backward()
            sgd(model,0.03)


        Yhat_test = model(data.Xtest)
        Y_test=data.Ytest.argmax(1)
        Ltest, acctest = loss_accuracy(Yhat_test,Y_test,loss)
        acctests.append(acctest)

    print ('max accuracy on test set: ' + str(max(acctests)))
