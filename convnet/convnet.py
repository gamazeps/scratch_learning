"""
Let's say the model is the same as http://scs.ryerson.ca/~aharley/vis/conv/

data:
    MNIST: 28x28px grayscale (8bits)

that is:

    16, 5x5 convolutions, padding 2, stride 1 -> 28x28x16
    max pooling 2x2 stride 2 -> 14x14x16
    16, 5x5 convolutions, padding 2, stride 1 -> 14x14x16
    max pooling 2x2 stride 2 7x7x16
    2 FC layer 784 -> 784 -> 10
    softmax classifier


"""
import numpy as np

class ConvLayer():
    def __init__(self, k, l, depth, stride, padding):
        self.k = k
        self.l = l
        self.depth = depth
        self.stride = stride
        self.padding = padding
        self.filters = np.random.randn(k, depth, l, l)
        self.biases = np.zeros((k, 1))

    def forward(self, X):
        assert(X.shape[0] == self.depth) # make sure that we have the proper input depth.
        assert(X.shape[1] == X.shape[2]) # make sure that the image is a square.
        out = int((X.shape[1] + 2*self.padding - self.l) / self.stride) + 1
        z = np.zeros((self.k, out, out))

        print(X.shape)
        print(z.shape)
        for k in range(0, self.k):
            for i in range(-self.padding, X.shape[1] + self.padding - self.l + 1, self.stride):
                for j in range(-self.padding, X.shape[2] + self.padding - self.l + 1, self.stride):
                    for k_d in range(0, self.depth):
                        for k_i in range(0, self.l):
                            for k_j in range(0, self.l):
                                if (i + k_i) >= 0 and (i + k_i) < X.shape[1] and (j + k_j) >= 0 and (j + k_j) < X.shape[2]:
                                    z[k][int((i)/self.stride)][int(j/self.stride)] += self.filters[k][k_d][k_i][k_j] * X[k_d][i + k_i][j + k_j]
                                # many errors there
                    z[k][i][j] += self.biases[k]

        return z

    def backward(self, dz, X):
        assert(dz.shape[0] == self.k) # make sure that we have the proper input depth.
        out = (X.shape[1] + 2*self.padding -self.l) / self.stride + 1
        assert(dz.shape[1] == out) # make sure that dz is of the right shape
        assert(dz.shape[2] == out) # make sure that dz is of the right shape

        dw = np.zeros((self.k, self.depth, self.l, self.l))
        db = np.zeros((self.k, 1))

        for k in range(0, self.k):
            for k_d in range(0, self.depth):
                for k_i in range(0, self.l):
                    for k_j in range(0, self.l):
                        for i in range(0, dz.shape[1]):
                            for j in range(0, dz.shape[2]):
                                if ((i * self.stride + k_i) >= 0 and
                                   (i * self.stride + k_i) < X.shape[1] and
                                   (j * self.stride + k_j) >= 0 and
                                   (j * self.stride + k_j) < X.shape[2]):
                                    dw[k][k_d][k_i][k_j] += dz[k][i][j] * X[k_d][i * self.stride + k_i][j * self.stride + k_j] 

        for k in range(0, self.k):
            for i in range(0, dz.shape[1]):
                for j in range(0, dz.shape[2]):
                    db += dz[k][i][j]

        dz_prev = None
        return {"dz": dz_prev, "dw": dw, "db": db}

class FullyConnected():
    def __init__(self, n_a, n_z):
        self.n_a = n_a
        self.n_z = n_z
        self.b = np.zeros((n_z, 1))
        self.W = np.random.randn(n_z, n_a) * 0.01

    def forward(self, A):
        return np.dot(self.W, A) + self.b

    def backward(self, dz, X):
        db = dz
        dw = np.dot(dz, X.transpose())
        dz = np.dot(self.W.transpose(), dz)
        return {"dz": dz_prev, "dw": dw, "db": db}

class ReLU():
    def __init__(self):
        pass

    def forward(self, A):
        return np.maximum(A, 0)

    def backwards(self, dz, X):
        return np.maximum(X, 0) * dz

    def update_params(self, dw, db):
        pass


class Model():
    def init(layers):
        pass

def main():
    conv = ConvLayer(k=1, l=3, depth=1, stride=1, padding=1)
    X = np.random.rand(1, 3, 3)
    print(X)
    activation = conv.forward(X)
    print("activation")
    print(activation)

    print("backward")
    activation = conv.backward(X, activation)

main()
