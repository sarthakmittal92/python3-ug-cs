import torch
from torch import nn, optim
from torch.nn import functional as F

# linear regression network
class Network(nn.Module):
    
    # initialise
    def __init__(self, nft, nhid, nout):
        super().__init__()
        self.hidden = nn.Linear(nft, nhid)
        self.predict = nn.Linear(nhid, nout)
    
    # forward
    def forward(self, x):
        x = F.relu(self.hidden(x))      
        x = self.predict(x)      
        return x
    
    # train
    def train(self, X, y, epochs, lr):
        optimizer = optim.SGD(self.parameters(), lr = lr)
        total_loss = 0
        print(f'Training for {epochs} epochs..')
        for epoch in range(epochs):
            pred = self.forward(X)
            loss = torch.norm(pred - y.T) ** 2 / len(pred)
            self.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f'Epoch: {epoch}, Loss: {total_loss}')
        preds = self.forward(X)
        return preds

# embedding network
class SetEmbed(nn.Module):
    
    # initialise
    def __init__(self, nft, nout, k1 = 16, k2 = 12):
        super().__init__()
        self.init_layer = [
            nn.Linear(nft, k1),
            nn.ReLU(),
            nn.Linear(k1, k2)
        ]
        self.init_layer = nn.Sequential(*self.init_layer)
        self.net = [
            nn.Linear(k2, nout)
        ]
        self.net = nn.Sequential(*self.net)
    
    # forward
    def forward(self, X, setSizes = None):
        # input: X (batch-size * max-set-size * embed-dim)
        assert len(X.shape) == 3
        assert(X.shape[0] == len(setSizes))
        X = self.init_layer(X)
        N = len(X)
        padded = torch.zeros(X.shape)
        for i in range(N):
            padded[i][:setSizes[i]] = X[i][:setSizes[i]]
        X = padded
        X = torch.sum(X, dim = 1)
        X = self.net(X)
        return X

# pairwise ranking loss with query ID
def pwRankingLoss(predPos, predNeg, qidPos, qidNeg):
    return (nn.ReLU()(predNeg[:,None] - predPos[None,:]) * (qidNeg[:,None] == qidPos[None,:])).sum() / (qidPos[:,None] == qidNeg[None,:]).sum()

# subset selection
def subSelect(X, S):
    # input: X (n * d), list of tensors of indices
    output = None
    P = torch.matmul(X,X.T)
    output = []
    for s in S:
        output.append(P[s,:][:,s])
    return output