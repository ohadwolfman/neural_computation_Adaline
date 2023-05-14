import itertools
import os
import pandas as pd
import numpy as np
from collections import namedtuple

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import chardet

print(" -----loading the files------- ")
folder1_path = 'קבצים (מ,ב,ל)'
file1 = 'Data_regular_writing.txt'
file_names1 = os.listdir(folder1_path)
print("first folder loaded")

folder2_path = 'קבצים בהטייה של 15_ לשני הצדדים (מ,ב,ל)'
file2 = 'Data_15%_rotated_writing.txt'
file_names2 = os.listdir(folder2_path)
print("second folder loaded")


print("\n -----Convert the file's content into DataFrame------- ")
def read_and_create_df(fila_name):
    with open(fila_name, 'r', encoding='latin-1') as f:
        content = f.readlines()

    # remove newline characters and split each line into a list of values
    content = [s.replace(', ', ',').replace(' ,', ',').replace(']', '').replace('[', '') for s in content]
    content = [line.strip().strip('()').split(',') for line in content]
    df = pd.DataFrame(content)
    return df

# If we would want to create the DataFrame from folder that contains txt files of vectors we would use this function:
def read_and_process_files(folder_path):
    file_names = os.listdir(folder_path)
    fulldf = pd.DataFrame()

    for file in file_names:
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r', encoding='latin-1') as f:
            content = f.readlines()

        # remove newline characters and split each line into a list of values
        content = [s.replace(', ', ',').replace(' ,', ',').replace(']', '').replace('[', '') for s in content]
        content = [line.strip().strip('()').split(',') for line in content]

        # create a dataframe from the list of values
        df = pd.DataFrame(content)
        fulldf = pd.concat([fulldf, df], ignore_index=True)
    return fulldf


# Join the DataFrames from both of the files
finaldf = pd.concat([read_and_create_df(file1),
                    read_and_create_df(file2)], ignore_index=True)

# Changing the first column to 'category' - that present the label column
finaldf.rename({0:'category'}, axis=1, inplace=True)

print(f"The full df has been loaded and contain {finaldf.shape[0]} rows")
print("Example of the data:")
print(finaldf.head(3))  # present 3 first rows


print("\n -----Clean the DataFrame------- ")
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


print("\n -----The data contains 3 types of Hebrew letters: ב,ל,מ ------- ")
print(" -----Creating 3 tables to Comparisons: (ב,ל),(ב,מ),(ל,מ) ------- ")
condition1 = finaldf['category'] == 1.0  # 1 represent the letter ב
condition2 = finaldf['category'] == 2.0  # 2 represent the letter ל
condition3 = finaldf['category'] == 3.0  # 3 represent the letter מ
data23 = finaldf[~condition1]
data13 = finaldf[~condition2]
data12 = finaldf[~condition3]
data23 = data23.reset_index().drop('index', axis=1)
data13 = data13.reset_index().drop('index', axis=1)
data12 = data12.reset_index().drop('index', axis=1)

print(f"Shape of data12: {data12.shape}")
print(f"Shape of data23: {data23.shape}")
print(f"Shape of data13: {data13.shape}")


# Adaline class
class Adline:
    def __init__(self, lr=0.00001, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        self.weights = np.zeros(1 + X.shape[1]).reshape(-1, 1)

        # self.weights = np.random.rand(X.shape[1]+1, 1)
        # self.weights = np.full((101,1),0.001)
        self.errors = []

        for i in range(self.n_iter):
            output = self.activation_function(self.net_input(X))
            Y = y.values
            output = output.reshape(-1, 1)
            errors = Y - output

            self.weights[1:] = self.weights[1:] + self.lr * X.T.dot(errors)
            self.weights[0] = self.weights[0] + self.lr * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.errors.append(cost)

            if i > 2 and np.isclose(self.errors[-2], self.errors[-1], rtol=1e-4):
                return i, errors
                break
        return self

    def net_input(self, X):
        return np.dot(X.astype('float64'), self.weights[1:]) + self.weights[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def get_params(deep=True):
        """
        Get parameters of the Adline model.
        """
        return {
            'lr': self.lr,
            'n_iter': self.n_iter,
            'weights': self.weights
        }

    def cross_validate(self, X, y, n_folds=10):
        """
        Perform n-fold cross-validation for the Adline model on the given data.
        """
        kf = KFold(n_splits=n_folds)
        scores = []
        iters = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index[0]:], X[test_index[0]:]
            y_train, y_test = y[train_index[0]:], y[test_index[0]:]
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
            (i, cost) = self.fit(X_train, y_train)
            iters.append(i)
            score = self.score(X_test, y_test)
            scores.append(score)

        return scores, np.mean(scores), iters, cost

    def score(self, X, y):
        """
        Return the accuracy score for the Adline model on the given data.
        """
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        return accuracy

    def activation_function(self, X):
        return X


def trainSubTable(df):
    X = df.drop(columns=['category'])
    y = pd.DataFrame(df.category)
    if (df['category'] == 3.0).any():
        y = y.replace(3.0, -1.0)
    elif (df['category'] == 2.0).any():
        y = y.replace(2.0, -1.0)
    if (df['category'] == 2.0).any() and (df['category'] == 3.0).any():
        y = y.replace(2.0, 1.0)

    # Scale features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    adaline = Adline()

    scores, scores_mean, i, cost = adaline.cross_validate(X_train, y_train, n_folds=5)
    final_scores = adaline.score(X_test, y_test)

    d = {'score': final_scores}
    return d, scores, scores_mean, i, cost


print("\n -----Calculating the datasets ------- ")
test_score12,scores12,scores_mean12,i12,cost12 = trainSubTable(data12)
test_score13,scores13,scores_mean13,i13,cost13 = trainSubTable(data13)
test_score23,scores23,scores_mean23,i23,cost23 = trainSubTable(data23)

# Calculate the accuracy of the model on the test set
std_dev12 = np.std(scores12)
std_dev13 = np.std(scores13)
std_dev23 = np.std(scores23)

print("\n -----Predicting by Adaline algorithm 1(ב) versus 2(ל) ------- ")
print("Std accuracy scores: ", std_dev12)
print("Average number of iterations before convergence:", i12, " Mean:", np.mean(i12))
print("accuracies in the cross validation:", scores12)
print("Average accuracy across all folds:", scores_mean12)
print("Test accuracy score:", test_score12)


print("\n -----Predicting by Adaline algorithm 1(ב) versus 3(מ) ------- ")
print("Std accuracy scores: ", std_dev13)
print("Average number of iterations before convergence:", i13, " Mean:" , np.mean(i13))
print("accuracies in the cross validation:", scores13)
print("Average accuracy across all folds:", scores_mean13)
print("Test accuracy score:", test_score13)


print("\n -----Predicting by Adaline algorithm 2(ל) versus 3(מ) ------- ")
print("Std accuracy scores: ", std_dev23)
print("Average number of iterations before convergence:", i23, " Mean:" , np.mean(i23))
print("accuracies in the cross validation:", scores23)
print("Average accuracy across all folds:", scores_mean23)
print("Test accuracy score:", test_score23)