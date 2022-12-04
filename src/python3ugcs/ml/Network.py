import numpy as np
import math

# flatten layer
class FlattenLayer:
    
    # initialise
    def __init__(self, inpSh):
        self.inpSh = inpSh
    
    # forward
    def forward(self, input):
        input = input.flatten()
        return input
    
    # backward
    def backward(self, outErr, lr):
        outErr = outErr.reshape(self.inpSh)
        return outErr

# fully connected layer
class FCLayer:
    
    # initialise
    def __init__(self, inpSz, outSz):
        self.inpSz = inpSz
        self.outSz = outSz
        v = math.sqrt(6) / (self.input_size + self.output_size)
        self.weights = -v + np.random.rand(self.input_size,self.output_size) * (2 * v)
        self.bias = np.random.rand(1,self.output_size)
    
    # forward
    def forward(self, input):
        self.input = input
        output = input @ self.weights + self.bias
        return output
    
    # backward
    def backward(self, outErr, lr):
        grad = np.reshape(outErr,(1,-1))
        input = np.squeeze(self.input)
        self.weights -= lr / self.inpSz * input[None,:].T @ grad
        self.bias -= lr / self.inpSz * grad
        return grad @ self.weights.T

# activation layer
class ActivationLayer:
    
    # initialise
    def __init__(self, activ, activ_p):
        self.activ = activ
        self.activ_p = activ_p
    
    # forward
    def forward(self, input):
        output = self.activ(input)
        self.input = input
        return output
    
    # backward
    def backward(self, outErr, lr):
        grad = self.activ_p(self.input) * outErr
        return grad

# softmax layer
class SoftmaxLayer:
    
    # initialise
    def __init__(self, inpSz):
        self.inpSz = inpSz
    
    # forward
    def forward(self, input):
        input = np.clip(input, -10, 10)
        output = np.exp(input) / np.sum(np.exp(input))
        self.output = output
        self.input = input
        return output
    
    # backward
    def backward(self, outErr, lr):
        sm = self.output
        J = sm * np.identity(sm.size) - sm.T @ sm
        grad = outErr[None,:] @ J
        return grad

# activators
class Activators:
    
    def __init__(self):
        pass
    
    def sigmoid(self, x):
        x = np.clip(x, -100, 100)
        y = 1 / (1 + np.exp(-x))
        return y
    
    def sigmoid_prime(self, x):
        s = self.sigmoid(x)
        y = s * (1 - s)
        return y

    def tanh(self, x):
        y = np.tanh(x)
        return y

    def tanh_prime(self, x):
        y = 1 - self.tanh(x) ** 2
        return y

    def relu(self, x):
        y = np.maximum(x,0)
        return y

    def relu_prime(x):
        y = (x > 0) * 1
        return y

# classification network
class ClassNet:
    
    # initialise
    def __init__(self, data):
        self.A = Activators()
        if data == 'mnist':
            self.network = [
                FlattenLayer(input_shape=(28, 28)),
                FCLayer(28 * 28, 12),
                ActivationLayer(self.A.sigmoid, self.A.sigmoid_prime),
                FCLayer(12, 10),
                SoftmaxLayer(10)
            ]
            self.epochs = 10
            self.lr = 0.1
        else:
            self.network = [
                FlattenLayer(input_shape=2048),
                FCLayer(2048, 12),
                ActivationLayer(self.A.sigmoid, self.A.sigmoid_prime),
                FCLayer(12, 5),
                SoftmaxLayer(5)
            ]
            self.epochs = 50
            self.lr = 0.1
    
    # cross entropy
    def crossEnt(self, y_true, y_pred):
        loss = -math.log(y_pred[0][y_true])
        return loss
    
    # cross entropy prime
    def crossEntPr(self, y_true, y_pred):
        grad = np.zeros(y_pred.shape[1])
        grad[y_true] -= 1 / y_pred[0][y_true]
        return grad
    
    # pre-process
    def preprocessing(self, X):
        m = np.mean(X)
        s = np.std(X)
        X_out = (X - m) / s
        assert X_out.shape == X.shape
        return X_out
    
    # split
    def split(self, X, Y, train_ratio=0.8):
        X_transformed = self.preprocessing(X)
        assert X_transformed.shape == X.shape
        num_samples = len(X)
        indices = np.arange(num_samples)
        num_train_samples = math.floor(num_samples * train_ratio)
        train_indices = np.random.choice(indices, num_train_samples, replace=False)
        val_indices = list(set(indices) - set(train_indices))
        X_train, Y_train, X_val, Y_val = X_transformed[train_indices], Y[train_indices], X_transformed[val_indices], Y[val_indices]
        return X_train, Y_train, X_val, Y_val
    
    # train
    def train(self, X_train, Y_train):
        print(f'Training for {self.epochs} epochs..')
        for epoch in range(self.epochs):
            error = 0
            for x, y_true in zip(X_train, Y_train):
                output = x
                for layer in self.network:
                    output = layer.forward(output)
                error += self.crossEnt(y_true, output)
                outErr = self.crossEntPr(y_true, output)
                for layer in reversed(self.network):
                    outErr = layer.backward(outErr, self.lr)
            error /= len(X_train)
            print(f'Epoch: {epoch}, Error: {error}')
    
    # predict
    def predict(self, X_test):
        Y_test = np.zeros(X_test.shape[0])
        X_test = self.preprocessing(X_test)
        i = 0
        for x in X_test:
            output = x
            for layer in self.network:
                output = layer.forward(output)
            Y_test[i] = np.argmax(output)
            i += 1
        assert Y_test.shape == (X_test.shape[0],) and type(Y_test) == type(X_test), "Check what you return"
        return Y_test