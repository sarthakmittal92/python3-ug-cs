import torch
from torch import nn
import numpy as np
import pandas as pd

# regression model
class RegrModel(nn.Module):
    
    # initialise
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(2,1).double())
        self.b1 = nn.Parameter(torch.randn(1).double())
        self.w2 = nn.Parameter(torch.randn(1,1).double())
        self.b2 = nn.Parameter(torch.randn(1).double())
    
    # forward
    def forward(self, X):
        pred = self.w2 * torch.tanh(self.w1[0] * X + self.w1[1] * (X ** 2) + self.b1) + self.b2
        return pred.squeeze()
    
    # L1 loss
    def l1(self, X, Y, w):
        loss = (1 / len(X)) * torch.sum(abs(Y - torch.matmul(X,w)),0)
        return (loss.item())
    
    # L2 loss
    def l2(self, X, Y, w):
        loss = (1 / len(X)) * torch.sum((Y - torch.matmul(X,w)) ** 2,0)
        return (loss.item())
    
    # L2 loss derivative
    def l2grad(X, Y, w):
        deriv = (1 / len(X)) * torch.matmul(torch.t(X),(2 * (torch.matmul(X,w) - Y)))
        return deriv
    
    # weights
    def weights(self, X, Y):
        # https://tinyurl.com/24yhermw
        # https://tinyurl.com/2bhnjxu4
        w_closed = torch.round(torch.matmul(torch.inverse((torch.matmul(torch.t(X),X))),torch.matmul(torch.t(X),Y)), decimals = 4)
        w_closed = w_closed.detach().squeeze().numpy()
        return w_closed
    
    # split
    def split(self, dataframe):
        total_samples = dataframe.shape[0]
        train_ratio = .8
        random_indices = np.random.permutation(total_samples)
        train_set_size = int(train_ratio * total_samples)
        train_indices = random_indices[:train_set_size]
        test_indices = random_indices[train_set_size:]
        return dataframe.iloc[train_indices], dataframe.iloc[test_indices]
    
    # data generate
    def dataGen(self):
        data = pd.read_csv('generated_data.csv')
        np.random.seed(20)
        total_samples = data.shape[0]
        train_ratio = .8
        random_indices = np.random.permutation(total_samples)
        train_set_size = int(train_ratio*total_samples)
        train_indices =  random_indices[:train_set_size]
        test_indices = random_indices[train_set_size:]
        data.iloc[train_indices], data.iloc[test_indices] 
        X_train = (data.iloc[train_indices].iloc[:,:-1]).to_numpy()     # Design matrix for train data 
        y_train = (data.iloc[train_indices].iloc[:,-1]).to_numpy()      # Labels for train data
        y_train = y_train.reshape((y_train.shape[0],1))
        X_test = (data.iloc[test_indices].iloc[:,:-1]).to_numpy()       # Design matrix for test data
        y_test = (data.iloc[test_indices].iloc[:,-1]).to_numpy()        # Labels for test data
        y_test = y_test.reshape((y_test.shape[0],1))
        return {'X_train': X_train, 'Y_train':y_train, 'X_test': X_test, 'Y_test': y_test}
    
    # create weights
    def createW(self, data_dictionary, lambda_val):
        # https://tinyurl.com/24yhermw
        X = data_dictionary['X_train']
        _, d = X.shape
        Y = data_dictionary['Y_train']
        weights = np.matmul(np.linalg.inv((np.matmul(X.T,X)) + lambda_val * np.eye(d)),np.matmul(X.T,Y))
        return weights
    
    # training model
    def train_model(self, X_train, Y_train, X_test, Y_test):
        d = X_train.size(dim=1)  # No of features
        w = torch.randn(d, 1).double()  # initialize weights
        epsilon = 1e-15  # Stopping precision
        eta = 1e-3  # learning rate
        old_loss = 0
        epochs = 0  # No of times w updates
        test_err = []  # Initially empty list
        while (abs(self.l2(X_train, Y_train, w) - old_loss) > epsilon):
            old_loss = self.l2(X_train, Y_train, w)  # compute loss
            dw = self.l2grad(X_train, Y_train, w)  # compute derivate
            w = w - eta * dw  # move in the opposite direction of the derivate
            epochs += 1
            test_err.append(self.l2(X_test, Y_test, w))
        return w, epochs, test_err
    
    # test error
    def genTestErr(self, data_dictionary, weights):
        X = data_dictionary['X_test']
        n, _ = X.shape
        Y = data_dictionary['Y_test']
        test_error = (1 / n) * np.sum((Y - np.matmul(X,weights)) ** 2,0)
        return test_error
    
    # train
    def train(self, X, Y, model, loss_fn, optim, max_iter):
        loss_list = []
        for epoch in range(max_iter):
            Y_pred = model(X)
            loss = loss_fn(Y_pred, Y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            if epoch % 1000 == 0:
                loss_list += [loss.data.item()]
        return loss_list

# logistic regression
class LogiReg:
    
    # initialise
    def __init__(self, xdim, ydim, args):
        self.xdim = xdim
        self.ydim = ydim
        self.args = args
        self.weights, self.bias = self.init_weights(xdim, ydim)
        assert self.weights.shape == (xdim, )
    
    # initialise weights
    def init_weights(self, xdim):
        w = torch.zeros((xdim))
        b = 0
        return w, b
    
    # forward
    def forward(self, batch_x):
        assert (len(batch_x.size())==2)
        yhat = torch.sigmoid(self.args.temp * (torch.matmul(batch_x,self.weights) + self.bias))
        assert yhat.shape == (batch_x.shape[0], )
        return yhat
    
    # backward
    def backward(self, batch_x, batch_y, batch_yhat):
        assert (len(batch_x.size())==2)
        assert (len(batch_y.size())==1)
        assert (len(batch_yhat.size())==1)
        # https://towardsdatascience.com/an-overview-of-the-gradient-descent-algorithm-8645c9e4de1e
        N = len(batch_y)
        weights_new = self.weights - (1 / N) * self.args.lr * self.args.temp * torch.matmul(torch.t(batch_x),batch_yhat - batch_y)
        bias_new = self.bias - (1 / N) * self.args.lr * self.args.temp * torch.sum(batch_yhat - batch_y,0)
        self.weights = weights_new
        self.bias = bias_new
        return weights_new, bias_new
    
    # loss
    def loss(self, y, y_hat):
        assert (len(y.size())==1)
        assert (len(y_hat.size())==1)
        N = len(y_hat)
        one = torch.ones((N))
        loss = (-1 / N) * torch.sum(torch.mul(y,torch.log(y_hat)) + torch.mul(one - y,torch.log(one - y_hat)))
        assert len(loss.shape) == 0, "loss should be a torch scalar"
        return loss
    
    # score
    def score(self, yhat, y):
        assert (len(y.size())==1)
        assert (len(yhat.size())==1)
        yhat = yhat > 0.5
        return torch.sum(yhat == y)/float(y.shape[0])

# linear regression
class LinReg:
    
    # initialise
    def __init__(self, xdim, args):
        self.xdim = xdim
        self.args = args 
        self.weights, self.bias = self.init_weights(xdim)
        assert self.weights.shape == (xdim, )
    
    # initialise weights
    def init_weights(self, xdim):
        w = torch.zeros((xdim))
        b = 0
        return w, b
    
    # forward
    def forward(self, batch_x):
        assert (len(batch_x.size())==2)
        yhat = torch.matmul(batch_x,self.weights) + self.bias
        assert yhat.shape == (batch_x.shape[0], )
        return yhat
    
    # backward
    def backward(self, batch_x, batch_y, batch_yhat):
        assert (len(batch_x.size())==2)
        assert (len(batch_y.size())==1)
        assert (len(batch_yhat.size())==1)
        # https://towardsdatascience.com/an-overview-of-the-gradient-descent-algorithm-8645c9e4de1e
        N = len(batch_y)
        weights_new = self.weights - (2 / N) * self.args.lr * torch.matmul(torch.t(batch_x),batch_yhat - batch_y)
        bias_new = self.bias - (2 / N) * self.args.lr * torch.sum(batch_yhat - batch_y,0)
        self.weights = weights_new
        self.bias = bias_new
        return weights_new, bias_new
    
    # loss
    def loss(self, y, y_hat):
        assert (len(y.size())==1)
        assert (len(y_hat.size())==1)
        loss = torch.mean(torch.square(y - y_hat))
        assert len(loss.shape) == 0, "loss should be a torch scalar"
        return loss
    
    # score
    def score(self, yhat, y):
        assert (len(y.size())==1)
        assert (len(yhat.size())==1)
        return - torch.mean(torch.square(y-yhat))

# early stopping
class EarlyStop(object):
    
    # initialise
    def __init__(self, args):
        self.args = args
        self.patience = args.patience 
        self.delta = args.delta
        self.best_score = None
        self.num_bad_epochs = 0 
        self.should_stop_now = False
    
    # check
    def check(self, curr_score) :
        if self.best_score == None or curr_score > self.best_score:
            self.best_score = curr_score
        else:
            self.num_bad_epochs += 1
        if self.num_bad_epochs > self.patience:
            self.should_stop_now = True
        return self.should_stop_now

# mini batch
def minibatch(trn_X, trn_y, batch_size):
    # https://stackoverflow.com/questions/44738273/torch-how-to-shuffle-a-tensor-by-its-rows
    N = len(trn_X)
    indices = torch.randperm(N)
    trn_X = trn_X[indices]
    trn_y = trn_y[indices]
    for i in range(0,N,batch_size):
        if i + batch_size <= N:
            batch_x = trn_X[i:i + batch_size]
            batch_y = trn_y[i:i + batch_size]
        else:
            batch_x = trn_X[i:N]
            batch_y = trn_y[i:N]
        yield (batch_x, batch_y)

# split data
def split_data(X:torch.Tensor, y:torch.Tensor, split_per = 0.6):
    shuffled_idxs = torch.randperm(X.shape[0])
    trn_idxs = shuffled_idxs[0:int(split_per * X.shape[0])]
    tst_idxs = shuffled_idxs[int(split_per * X.shape[0]):]
    return X[trn_idxs], y[trn_idxs], X[tst_idxs], y[tst_idxs]

# train
def train(args, X_tr, y_tr, X_val, y_val, model):
    es = EarlyStop(args)
    losses = []
    val_acc = []
    epoch_num = 0
    print(f'Training for {args.max_epochs} epochs..')
    while (epoch_num <= args.max_epochs): 
        for idx, (batch_x, batch_y) in enumerate(minibatch(X_tr, y_tr, args.batch_size)):
            if idx == 0:
                assert batch_x.shape[0] == args.batch_size
            batch_yhat = model.forward(batch_x)
            losses.append(model.loss(batch_y, batch_yhat).item())
            _ = model.backward(batch_x, batch_y, batch_yhat)
        val_score = model.score(model.forward(torch.Tensor(X_val)), torch.Tensor(y_val))
        print(f'Epoch: {epoch_num}, Validation Score: {val_score}')
        val_acc.append(val_score)
        if es.check(val_score,model,epoch_num):
            break
        epoch_num +=1  
    return losses, val_acc