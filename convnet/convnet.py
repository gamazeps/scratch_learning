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

conv1_k = 16
conv1_l = 5
conv2_k = 16
conv2_l = 5
n_classes = 10
input_w = 28

def init_params():
    """
    Conv size is hardcoded and this sis bad
    """
    params = dict()
    params["conv1_W"] = np.random((conv1_k, conv1_l, conv1_l)) * 0.01
    params["conv1_b"] = np.zeros((conv1_k, 1))
    params["conv2_W"] = np.random((conv2_k, conv2_l, conv2_l)) * 0.01
    params["conv2_b"] = np.zeros((conv2_k, 1)) 
    fc_size = ((input_w / 4) ^ 2) * conv2_k # flatten data after max pooling.
    params["W1"] = np.random((fc_size, fc_size)) * 0.01
    params["b1"] = np.zero((fc_size, 1))
    params["W2"] = np.random((n_classes, fc_size)) * 0.01
    params["b2"] = np.zero((n_classes, 1))
    return params

def forward(params, X):
    pass
    
def main():
    pass

if __name__ is "__main__":
    main()
