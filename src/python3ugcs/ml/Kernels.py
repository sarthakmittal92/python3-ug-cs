import numpy as np
import math
from matplotlib.image import imread

# kernel regression model
class KernelRegr:
    
    # initialise
    def __init__(self, x, y, b = 1):
        self.x = x
        self.y = y
        self.b = b
    
    # Gaussian kernel
    def RBF(self, z):
        return np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
    
    # predict
    def predict(self, x_test):
        w = np.squeeze(self.RBF(np.subtract.outer(self.x,x_test) / self.b))
        w /= np.sum(w,0)
        return np.matmul(w.T,self.y)

# kernel methods module
class KernelMethods:
    
    # initialise
    def __init__(self):
        pass
    
    # poly-features
    def polyft(self, x):
        xprime = np.array([
            x[:,0],
            x[:,1],
            x[:,0] ** 3,
            x[:,1] ** 3
        ]).T
        return xprime

    # poly kernel
    def polyK(self, X, Y):
        xf = self.polyft(X)
        yf = self.polyft(Y)
        kernel = np.matmul(xf,yf.T)
        return kernel

    # two dimensional RBF
    def twoDimRBF(self, a, b):
        # input: a (n * d), b (n * d)
        n, m = len(a), len(b)
        a1 = np.tile(np.linalg.norm(a, axis = 1),(m,1)).T
        b1 = np.tile(np.linalg.norm(b, axis = 1),(n,1))
        kernel = np.exp(-(a1 + b1 - 2 * np.matmul(a,b.T)) / 2)
        return kernel
    
    # weighted convolution
    def wConv(self, x, w, b, params):
        # input: x (n * c * h * w), w (f * c * hh * ww), b (f * 1)
        # output: convolved (N, F, 1 + (H + 2 * pad - HH) / stride, 1 + (W + 2 * pad - WW) / stride)
        padding, stride = params['pad'], params['stride']
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        H1 = math.floor(1 + (H + (2 * padding) - HH) / stride)
        W1 = math.floor(1 + (W + (2 * padding) - WW) / stride)
        x = np.pad(x,((0,0),(0,0),(padding,padding),(padding,padding)), mode = 'constant')
        out = np.zeros((N,F,H1,W1))
        for i in range(H1):
            for j in range(W1):
                a = x[:,:,(i * stride):((i * stride) + HH),(j * stride):((j * stride) + WW)]
                a = np.reshape(a,(N,C * HH * WW))
                w = np.reshape(w,(F,C * HH * WW))
                out[:,:,i,j] = np.matmul(a,w.T) + b
        cache = (x, w, b, params)
        return out, cache
    
    # gram matrix
    def gram(self, x):
        H, W, C = x.shape
        a = np.reshape(x,(H * W,C))
        out = np.matmul(a.T,a)
        return out
    
    # Sobel filter
    def sobelFilter(self, img):
        image_file = img
        input_image = imread(image_file)  # this is the array representation of the input image
        r_img, g_img, b_img = input_image[:, :, 0], input_image[:, :, 1], input_image[:, :, 2]
        gamma = 1.400  
        r_const, g_const, b_const = 0.2126, 0.7152, 0.0722  # weights for the RGB components respectively
        grayscale_image = r_const * r_img ** gamma + g_const * g_img ** gamma + b_const * b_img ** gamma
        Gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        [rows, columns] = np.shape(grayscale_image)  # we need to know the shape of the input grayscale image
        sobel_filtered_image = np.zeros(shape = (rows, columns))  # initialization of the output image array (all elements are 0)
        i = 0
        j = 0
        for _1 in range(rows)[:-2]:
            for _2 in range(columns)[:-2]:
                S1 = np.sum(np.multiply(Gx, grayscale_image[i:i + 3, j:j + 3]))
                S2 = np.sum(np.sum(np.multiply(Gy, grayscale_image[i:i + 3, j:j + 3])))
                sobel_filtered_image[i + 1, j + 1] = np.sqrt(np.sum([np.square(S1), np.square(S2)]))
                j += 1
            i += 1
            j = 0
        sobel_filtered_image *= 255.0 / sobel_filtered_image.max()
        assert type(sobel_filtered_image) == np.ndarray
        return sobel_filtered_image
    
    # Gaussian filter
    def GaussianFilter(self, x, w, stride):
        N, H, W = x.shape
        F, HH, WW = w.shape
        out = np.zeros((N,F,H,W))
        for i in range(H):
            for j in range(W):
                if i < HH // 2 or i >= H - (HH // 2) or j < WW // 2 or j >= W - (WW // 2):
                    out[:,:,i,j] = x[:,i,j]
                else:
                    a = x[:,i - HH // 2:i + HH // 2 + 1,j - WW // 2:j + WW // 2 + 1]
                    a = np.reshape(a,(N,HH * WW))
                    w = np.reshape(w,(F,HH * WW))
                    out[:,:,i,j] = math.ceil(np.matmul(a,w.T))
        return out

# kernel PCA module
class KernelPCA:
    
    # initialise
    def __init__(self):
        self.params = {
            'sigma_gauss': 1,
            'gamma_rbf': 1,
            'sigma_laplace': 1
        }    
    
    # RBF
    def RBF(self, a, b):
        sigma = self.params['sigma_gauss']
        n, m = len(a), len(b)
        a1 = np.tile(np.linalg.norm(a, axis = 1),(m,1)).T
        b1 = np.tile(np.linalg.norm(b, axis = 1),(n,1))
        out = np.exp(-(a1 ** 2 + b1 ** 2 - 2 * np.matmul(a,b.T)) / (2 * (sigma ** 2)))
        return out
    
    # RBF for SVM
    def RBFgamma(self, a, b):
        gamma = self.params['gamma_rbf']
        n, m = len(a), len(b)
        a1 = np.tile(np.linalg.norm(a, axis = 1),(m,1)).T
        b1 = np.tile(np.linalg.norm(b, axis = 1),(n,1))
        out = np.exp(-gamma * (a1 ** 2 + b1 ** 2 - 2 * np.matmul(a,b.T)))
        return out
    
    # Laplace kernel
    def laplaceK(self, a, b):
        sigma = self.params['sigma_laplace']
        a1 = a[:,None,:]
        b1 = b[None,:,:]
        out = np.exp(-(np.linalg.norm(a1 - b1, ord = 1, axis = 2)) / sigma)
        return out
    
    # kernel PCA
    def kernel_pca(self, X: np.ndarray, kernel: str):
        n = X.shape[0]
        K = np.zeros((n,n))
        if kernel == 'poly':
            d = 5
            K = (np.dot(X,X.T) + 1)**d
        elif kernel == 'rbf':
            gamma = 15
            norm = np.square(np.linalg.norm(X, axis = 1,keepdims = True)) +  np.square(np.linalg.norm(X.T, axis=0,keepdims=True)) - 2 * np.dot(X,X.T)
            K = np.exp(-norm*gamma)
        elif kernel == 'radial':
            newX = np.zeros(X.shape)
            newX[:,0] = np.sqrt(np.square(X[:,0]) + np.square(X[:,1]))
            newX[:,1] = np.arctan2(X[:,1],X[:,0])
            K = np.dot(newX,newX.T)
        K_centered = K - np.dot(np.ones(K.shape) / n ,K) - np.dot(K,np.ones(K.shape) / n) + np.dot(np.ones(K.shape) / n, np.dot(K,np.ones(K.shape) / n))
        w,v = np.linalg.eigh(K_centered)
        v_pca = v[:,-2:] * np.sqrt(w[-2:])
        temp = np.copy(v_pca[:, 0])
        v_pca[:, 0] = v_pca[:, 1]
        v_pca[:, 1] = temp
        return v_pca