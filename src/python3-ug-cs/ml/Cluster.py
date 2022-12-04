import numpy as np

# K-Means cluster model
class KMeans:
    
    # initialise
    def __init__(self, n, X):
        self.n = n
        self.clusters = {}
        self.means = {}
        self.X = X
    
    # update clusters
    def updateClusters(self):
        for i in range(self.n):
            self.clusters[i] = []
        for p in self.X:
            d = []
            for i in range(self.n):
                d.append(np.linalg.norm(p - self.means[i]))
            self.clusters[np.argmin(np.array(d))].append(p)
    
    # update means
    def updateMeans(self):
        for i in range(self.n):
            self.means[i] = np.mean(self.clusters[i], axis = 0)
    
    # set labels
    def setLabels(self):
        self.labels = []
        for p in self.X:
            d = []
            for i in range(self.n):
                d.append(np.linalg.norm(p - self.means[i]))
            self.labels[np.argmin(np.array(d))].append(p)
    
    # train
    def train(self, epochs):
        N = len(self.X)
        for i in range(self.n):
            self.clusters[i] = []
            idx = np.random.randint(0,N)
            self.means[i] = self.X[idx]
        print(f'Training for {epochs} epochs..')
        for epoch in range(epochs):
            self.updateClusters()
            self.updateMeans()
            print(f'Epoch {epoch} complete')
        self.setLabels()
        return self.labels