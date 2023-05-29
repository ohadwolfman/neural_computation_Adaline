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
            if len(finaldf.iloc[i, :]) < 100:
                rows_to_drop.append(i)
                continue

            finaldf.iloc[i, :] = finaldf.iloc[i, :] #.astype(np.float32)
        except ValueError:
            rows_to_drop.append(i)

    print(f"There are {len(rows_to_drop)} rows to drop")
    finaldf = finaldf.drop(rows_to_drop).reset_index(drop=True)
    print(f"After dropping, the df contain {finaldf.shape[0]} rows")
    return finaldf


df = cleanData(finaldf)
df['category'].astype(np.float32)

print("\n -----Building the Neural Network ------- ")
df = np.array(df,dtype=np.float32)
m, n = df.shape
X = df[:,1:]
Y = df[:,0]
temporary = finaldf.iloc[:,:5]
print(temporary.head())
print(temporary.info())

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=42, shuffle=True)
print(x_train.shape), print(y_train.shape), print(x_test.shape), print(y_test.shape)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
# Creating and training the MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(60, 60), activation='tanh', solver='adam',
                    learning_rate_init=0.001, random_state=42,validation_fraction=0.2)

scores = cross_val_score(mlp, x_train, y_train, cv=kf)
mlp.fit(x_train, y_train)
# Making predictions on the test set
y_pred = mlp.predict(x_test)

# Evaluating the accuracy of the classifier
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)
