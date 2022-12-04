import torch
from torch import nn, optim
from collections import OrderedDict

# multi-task learning model
class MultiTask(nn.Module):
    
    # initialise
    def __init__(self, d, k):
        super().__init__()
        self.phi = nn.Linear(d,k)
        self.bin = nn.Sequential(OrderedDict([
            ('sigmoid', nn.Sigmoid()),
            ('binary', nn.Linear(k,1)),
            ('sig2', nn.Sigmoid())
        ]))
        self.reg = nn.Sequential(OrderedDict([
            ('sigmoid', nn.Sigmoid()),
            ('regr', nn.Linear(k,1))
        ]))
        self.thres = 0.5
    
    # forward
    def forward(self, x):
        clfHead = self.bin(self.phi(x))
        regHead = self.reg(self.phi(x))
        return clfHead, regHead
    
    # accuracy
    def accuracy(self, preds:torch.Tensor, targets:torch.Tensor):
        acc = torch.sum(preds == targets).item() / len(targets)
        return acc
    
    # precision
    def precision(self, preds:torch.Tensor, targets:torch.Tensor):
        pairs = zip(targets,preds)
        tp = [p[0].item() + p[1].item() for p in pairs].count(2)
        precision = tp / torch.sum(preds == 1).item()
        return precision
    
    # recall
    def recall(self, preds:torch.Tensor, targets:torch.Tensor):
        pairs = zip(targets,preds)
        tp = [p[0].item() + p[1].item() for p in pairs].count(2)
        recall = tp / torch.sum(targets == 1).item()
        return recall
    
    # f1-score
    def f1_score(self, preds:torch.Tensor, targets:torch.Tensor):
        p, r = self.precision(preds,targets), self.recall(preds,targets)
        f1 = 2 * p * r / (p + r)
        return f1
    
    # mean squared error
    def MSE(self, preds:torch.Tensor, targets:torch.Tensor):
        mse = torch.sum((targets - preds) ** 2).item() / len(targets)
        return mse
    
    # mean absolute error
    def MAE(self, preds:torch.Tensor, targets:torch.Tensor):
        mae = torch.sum(torch.abs(targets - preds)).item() / len(targets)
        return mae
    
    # split
    def split(self, X:torch.Tensor, y1:torch.Tensor, y2:torch.Tensor, train_pc):
        torch.random.manual_seed(5)
        N, _ = X.shape
        n = int(train_pc * N)
        idx = torch.randperm(N)
        trainIdx = idx[:n]
        valIdx = idx[n:]
        X_trn, y1_trn, y2_trn, X_val, y1_val, y2_val = X[trainIdx], y1[trainIdx], y2[trainIdx], X[valIdx], y1[valIdx], y2[valIdx]
        assert X_trn.shape[0] + X_val.shape[0] == X.shape[0]
        return  X_trn, y1_trn, y2_trn, X_val, y1_val, y2_val
    
    # train
    def train(self, X_trn, y1_trn, y2_trn):
        _, d = X_trn.shape
        k = 10
        alpha = 4
        epochs = 50
        lr = 0.06
        model = MultiTask(d,k)
        binLoss = nn.BCELoss()
        regLoss = nn.L1Loss()
        optimizer = optim.SGD(model.parameters(), lr = lr)
        print(f'Training for {epochs} epochs..')
        for epoch in range(epochs):
            model.train()
            ttl = 0
            N = len(y1_trn)
            y1, y2 = torch.zeros(N), torch.zeros(N)
            optimizer.zero_grad()
            for i in range(N):
                x = X_trn[i]
                y1_pred, y2_pred = model(x)
                y1[i] = y1_pred
                y2[i] = y2_pred
            l1 = binLoss(y1,y1_trn)
            l2 = regLoss(y2,y2_trn)
            loss = l1 + alpha * l2
            loss.backward()
            optimizer.step()
            ttl += loss
            print(f'Epoch: {epoch}, Loss: {ttl}')
        return model
    
    # predict
    def predict(self, model:nn.Module, X_tst:torch.Tensor):
        N, _ = X_tst.shape
        y1_preds, y2_preds = torch.zeros(N), torch.zeros(N)
        for i in range(N):
            x = X_tst[i]
            y1_pred, y2_pred = model(x)
            if y1_pred < 0.5:
                y1_preds[i] = 0
            else:
                y1_preds[i] = 1
            y2_preds[i] = y2_pred
        assert len(y1_preds.shape) == 1 and len(y2_preds.shape) == 1
        assert y1_preds.shape[0] == X_tst.shape[0] and y2_preds.shape[0] == X_tst.shape[0]
        assert len(torch.where(y1_preds == 0)[0]) + len(torch.where(y1_preds == 1)[0]) == X_tst.shape[0], "y1_preds should only contain classification targets"
        return y1_preds, y2_preds