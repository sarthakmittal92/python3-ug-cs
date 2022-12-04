import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

# classifier model
class Network(nn.Module):
    
    # initialise
    def __init__(self, args, n):
        super().__init__()
        self.args = args
        self.n = n
        self.lin = nn.Linear(self.n, 1)
    
    # forward
    def forward(self, x):
        pred = None
        if self.args.model_type == "svm":
            # https://tinyurl.com/5f959z42
            pred = self.lin(x)
        elif self.args.model_type == "nll":
            pred = torch.sigmoid(self.lin(x))
        elif self.args.model_type == "ranking":
            pred = self.lin(x)
        else:
            raise NotImplementedError()
        return torch.squeeze(pred)
    
    # loss function
    def loss(self, pred, label):
        N = len(pred)
        if self.args.model_type == "svm":
            # https://tinyurl.com/yku7w95b
            zero = torch.zeros(pred.shape)
            one = torch.ones(pred.shape)
            # https://tinyurl.com/bdezpc7n
            label2 = torch.ones(label.shape)
            label2[label == 0] = -1
            loss = (1 / N) * torch.sum(torch.max(1 - torch.mul(pred,label2),zero))
        elif self.args.model_type == "nll":
            one = torch.ones(pred.shape)
            loss = (-1 / N) * torch.sum(torch.mul(label,torch.log(pred)) + torch.mul(one - label,torch.log(one - pred)))
        elif self.args.model_type == "ranking":
            # https://tinyurl.com/37wmkr83
            one = torch.ones(pred.shape)
            n = torch.mul(one - label,pred)
            n = n[n.nonzero()]
            p = torch.mul(label,pred)
            p = p[p.nonzero()]
            np = n.reshape(1,-1) - p
            loss = torch.sum(np[np > 0])
        else: 
            raise NotImplementedError()
        return loss

# train
def train(args, Xtrain, Ytrain, Xval, Yval, model):
    trainLoader = DataLoader(TensorDataset(Xtrain, Ytrain), batch_size = args.batch_size, shuffle = True)
    evalLoader = DataLoader(TensorDataset(Xval, Yval), batch_size = args.batch_size, shuffle = False)
    # build model
    opt = optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr = args.lr, weight_decay = args.weight_decay)
    losses = []
    val_accs = []
    print(f'Training for {args.epochs} epochs..')
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in trainLoader:
            opt.zero_grad()
            pred = model(batch[0])
            label = batch[1]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        losses.append(total_loss)
        val_acc = evaluate(evalLoader, model)
        val_accs.append(val_acc)
        print(f'Epoch: {epoch}, Loss: {total_loss}, Validation Accuracy: {val_acc}')
    return val_accs, losses

# evaluate score of the model
def evaluate(loader, model):
    # input: loader and model
    model.eval() # This enables the evaluation mode for the model
    eval_score = 0
    count = 0
    if model.args.model_type == "ranking":
        p = torch.Tensor()
        n = torch.Tensor()
    for data in loader:
        with torch.no_grad():
            pred = torch.squeeze(model(data[0]))
            label = data[1]
            actual = torch.ones(pred.shape)
            count += 1
            if model.args.model_type == "svm": 
                label2 = torch.ones(label.shape)
                label2[label == 0] = -1
                actual[pred < 0] = -1
                eval_score += torch.eq(label2,actual).sum().item() / len(label2)
            elif model.args.model_type == "nll":
                actual[pred <= 0.5] = 0
                eval_score += torch.eq(label,actual).sum().item() / len(label)
            elif model.args.model_type == "ranking":
                one = torch.ones(pred.shape)
                n = torch.cat((n,torch.squeeze(torch.mul(one - label,pred))))
                n = torch.squeeze(n[n != 0])
                p = torch.cat((p,torch.squeeze(torch.mul(label,pred))))
                p = torch.squeeze(p[p != 0])
            else: 
                raise NotImplementedError()
    if model.args.model_type == "ranking":
        pn = p.reshape(-1,1) - n
        pn = pn[pn > 0]
        eval_score = pn.size()[0]
    else:
        eval_score /= count
    return eval_score