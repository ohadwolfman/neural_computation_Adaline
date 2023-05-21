import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print(" -----loading the files------- ")
# from the folder 'קבצים (מ,ב,ל)' we generated united file:
file1 = 'Data_regular_writing.txt'

# from the folder 'קבצים בהטייה של 15_ לשני הצדדים (מ,ב,ל)' we generated united file:
file2 = 'Data_15%_rotated_writing.txt'

# Convert the file's content into DataFrame
def read_and_create_df(fila_name):
    with open(fila_name, 'r', encoding='latin-1') as f:
        content = f.readlines()

    # remove newline characters and split each line into a list of values
    content = [s.replace(', ', ',').replace(' ,', ',').replace(']', '').replace('[', '') for s in content]
    content = [line.strip().strip('()').split(',') for line in content]
    df = pd.DataFrame(content)
    return df


# Join the DataFrames from both of the files
finaldf = pd.concat([read_and_create_df(file1),
                    read_and_create_df(file2)], ignore_index=True)

# Changing the first column to 'category' - that present the label column
finaldf.rename({0:'category'}, axis=1, inplace=True)


print("\n -----DataFrame details ------- ")
print("First dataFrame has",finaldf.shape[0],"rows of", finaldf.shape[1],"pixels")
print(finaldf.head())

print("\n -----Clean the DataFrame------- ")
def cleanData(finaldf):
    # Drop the rows with null, and in addition every row that returns an exception will be dropped too
    rows_to_drop = finaldf[finaldf.isnull().any(axis=1)].index.tolist()

    for i in range(len(finaldf)):
        try:
            finaldf.iloc[i, :] = finaldf.iloc[i, :].astype('float64')
        except ValueError:
            rows_to_drop.append(i)

    print(f"There are {len(rows_to_drop)} rows to drop")
    finaldf = finaldf.drop(rows_to_drop).reset_index(drop=True)
    print(f"After dropping, the df contain {finaldf.shape[0]} rows")
    return finaldf


df = cleanData(finaldf)
print(df.head())

print("\n -----Building the Neural Network ------- ")
df = np.array(df)
m, n = df.shape
X = df[:,1:]
Y = df[:,0]

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.1, random_state=42, shuffle=True)
print(x_train.shape), print(y_train.shape), print(x_test.shape), print(y_test.shape)

x_train = x_train.T  # shape - 100,1870
x_test = x_test.T  # shape - 100,208

# initialize parameters
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


W1, b1, W2, b2 = gradient_descent(x_train, y_train, 500, 0.1)
