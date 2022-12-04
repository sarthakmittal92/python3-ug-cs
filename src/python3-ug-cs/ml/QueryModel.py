import torch
from torch import nn, optim
from sklearn.metrics import average_precision_score as aps

# query-corpus model
class QCModel(nn.Module):
    
    # initialise
    def __init__(self, inpft = 5, outft = 5, k1 = 16, k2 = 12, margin:float = 0.1, *args, **kwargs):
        super().__init__()
        self.inpft = inpft
        self.outft = outft
        self.margin = margin
        self.init_layer = [
            nn.Linear(self.inp_features, k1),
            nn.ReLU(),
            nn.Linear(k1, k2)
        ]
        self.init_layer = nn.Sequential(*self.init_layer)
        self.init_layer = self.init_layer.float()
        self.net = [
            nn.Linear(k2, self.out_features)
        ]
        self.net = nn.Sequential(*self.net)
    
    # set embedding
    def setEmbed(self, set_items:torch.Tensor):
        set_embed = torch.zeros(1, self.outft)
        set_items = torch.sum(torch.stack(set_items), axis = 0)
        x = self.init_layer(set_items.float())
        set_embed = self.net(x)[None,:]
        assert set_embed.shape == (1, self.outft)
        return set_embed
    
    # forward
    def forward(self, queries:list, corpus:list):
        query_emb = torch.cat([self.set_embed(entry) for entry in queries]).squeeze() 
        assert query_emb.shape == (len(queries), self.outft)
        corpus_emb = torch.cat([self.set_embed(entry) for entry in corpus]).squeeze()
        assert corpus_emb.shape == (len(corpus), self.outft)
        scores = self.score(queries_embed = query_emb, corpus_embed = corpus_emb)
        assert scores.shape == (len(queries), len(corpus))
        return scores
    
    def score(self, queries_embed:torch.Tensor, corpus_embed:torch.Tensor):
        scores = torch.zeros(queries_embed.shape[0], corpus_embed.shape[0])
        d = queries_embed.shape[1]
        sub = -nn.ReLU()(queries_embed.reshape(-1,1) - corpus_embed.reshape(1,-1))
        for i in range(queries_embed.shape[0]):
            for j in range(corpus_embed.shape[0]):
                scores[i][j] = torch.sum(torch.diag(sub[i * d:(i + 1) * d,j * d:(j + 1) * d]))
        assert scores.shape == (queries_embed.shape[0], corpus_embed.shape[0])
        return scores
    
    # loss
    def rankingLoss(self, scores: torch.Tensor, ground_truth:torch.Tensor, margin:float):
        loss = torch.tensor(0)
        ground_truth = torch.from_numpy(ground_truth)
        loss = torch.sum(torch.sum(torch.nn.ReLU()(scores * (1 - ground_truth) - scores * (ground_truth) + margin)))
        assert len(loss.shape) == 0
        return loss
    
    # average precision
    def avgP(scores:torch.Tensor, ground_truth:torch.Tensor):
        avg_precision = float(0)
        avg_precision = aps(ground_truth,scores)
        assert isinstance(avg_precision, float) == True
        return avg_precision
    
    # train
    def train(self, trn_queries, corpus, trn_ground_truth):
        epochs = 15
        lr = 0.005
        optimizer = optim.Adam(self.parameters(), lr = lr)
        loss = torch.Tensor(0)
        total_loss = 0
        print(f'Training for {epochs} epochs..')
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            scores = self.forward(trn_queries,corpus)
            loss = self.rankingLoss(scores,trn_ground_truth,self.margin)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f'Epoch: {epoch}, Loss: {total_loss}')