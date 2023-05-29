# initialize parameters
import numpy as np


def init_param():
    W1 = np.random.rand(3,x_train.shape[0]) - 0.5
    b1 = np.random.rand(3,1) - 0.5
    W2 = np.random.rand(3,3) - 0.5
    b2 = np.random.rand(3,1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0,Z)

def derivReLU(Z):
    np.where(Z > 0, 1, 0)

def softmax(Z):
    Z = np.array(Z, dtype=np.float64)  # Convert to numpy array
    Z_exp = np.exp(Z - np.max(Z))
    return Z_exp / np.sum(Z_exp, axis=0, keepdims=True)

def forward_propagation(W1, b1, W2, b2, X):
    Z1 = np.dot(W1,X) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    num_classes = int(np.max(Y)) + 1
    one_hot_Y = np.eye(num_classes, dtype=int)[Y.astype(int)].T
    return one_hot_Y


def back_propagation(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * derivReLU(Z1)
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def get_predictions(A2):
    return np.argmax(A2, axis=0)

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_param()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_propagation(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 50 == 0:
            print("iteration: ",i)
            print("accuracy: ", get_accuracy(get_predictions(A2), Y))
    return W1, b1, W2, b2


#W1, b1, W2, b2 = gradient_descent(x_train, y_train, 500, 0.1)
