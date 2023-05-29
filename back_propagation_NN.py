import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

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


def concatFilesToDf(file1, file2):
    # Join the DataFrames from both of the files
    return pd.concat([read_and_create_df(file1),
                      read_and_create_df(file2)], ignore_index=True)


finaldf = concatFilesToDf(file1, file2)

# Changing the first column to 'category' - that present the label column
finaldf.rename({0: 'category'}, axis=1, inplace=True)

print("\n -----DataFrame details ------- ")
print("First dataFrame has", finaldf.shape[0], "rows of", finaldf.shape[1], "pixels")
print(finaldf.head())


print("\n -----Clean the DataFrame------- ")
def cleanData(finaldf):
    # Drop the rows with null, and in addition every row that returns an exception will be dropped too
    rows_to_drop = finaldf[finaldf.isnull().any(axis=1)].index.tolist()

    for i in range(len(finaldf)):
        try:
            if len(finaldf.iloc[i, :]) < 100:
                rows_to_drop.append(i)
                continue

            finaldf.iloc[i, :] = finaldf.iloc[i, :]  # .astype(np.float32)
        except ValueError:
            rows_to_drop.append(i)

    print(f"There are {len(rows_to_drop)} rows to drop")
    finaldf = finaldf.drop(rows_to_drop).reset_index(drop=True)
    print(f"After dropping, the df contain {finaldf.shape[0]} rows")
    return finaldf


df = cleanData(finaldf)
df['category'].astype(np.float32)


print("\n -----The data contains 3 types of Hebrew letters: ב,ל,מ ------- ")
print(" -----Creating 3 tables to Comparisons: (ב,ל),(ב,מ),(ל,מ) ------- ")
condition1 = df['category'] == '1'  # 1 represent the letter ב
condition2 = df['category'] == '2'  # 2 represent the letter ל
condition3 = df['category'] == '3'  # 3 represent the letter מ
data23 = df[~condition1]
data13 = df[~condition2]
data12 = df[~condition3]
data23 = data23.reset_index().drop('index', axis=1)
data13 = data13.reset_index().drop('index', axis=1)
data12 = data12.reset_index().drop('index', axis=1)

print(f"Shape of data12: {data12.shape}")
print(f"Shape of data23: {data23.shape}")
print(f"Shape of data13: {data13.shape}")


print("\n -----Building the Neural Network ------- ")
def buildNeuralNetwork(df):

    if (df['category'] == '3').any() and (df['category'] == '2').any():
        print("----------Comparing Lamed vs Mem----------\n")
    elif (df['category'] == '1').any() and (df['category'] == '2').any():
        print("----------Comparing Lamed vs Bet----------\n")
    elif (df['category'] == '3').any() and (df['category'] == '1').any():
        print("----------Comparing Bet vs Mem----------\n")

    # df = np.array(df, dtype=np.float32)
    X = df.iloc[:, 1:]
    Y = df.iloc[:, 0]
    temporary = finaldf.iloc[:, :5]
    print(temporary.head())

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)
    # print(x_train.shape), print(y_train.shape), print(x_test.shape), print(y_test.shape)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    # Creating and training the MLP classifier
    mlp = MLPClassifier(hidden_layer_sizes=(100, 70, 100), activation='relu', solver='adam',
                        learning_rate_init=0.001, random_state=42, validation_fraction=0.2)

    scores = cross_val_score(mlp, x_train, y_train, cv=kf)
    mlp.fit(x_train, y_train)
    # Making predictions on the test set
    y_pred = mlp.predict(x_test)

    # Evaluating the accuracy of the classifier
    accuracy = np.mean(y_pred == y_test)
    print("Accuracy:", accuracy)


buildNeuralNetwork(data12)
buildNeuralNetwork(data13)
buildNeuralNetwork(data23)