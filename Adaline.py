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
import matplotlib.pyplot as plt

print(" -----loading the files------- ")
folder1_path = 'קבצים (מ,ב,ל)'
file_names1 = os.listdir(folder1_path)
print("first folder loaded")

folder2_path = 'קבצים בהטייה של 15_ לשני הצדדים (מ,ב,ל)'
file_names2 = os.listdir(folder2_path)
print("second folder loaded")


print("\n -----Convert the file's content into DataFrame------- ")
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


finaldf = pd.concat([read_and_process_files(folder1_path),
                    read_and_process_files(folder2_path)], ignore_index=True)
finaldf.rename({0:'category'}, axis=1, inplace=True)

print(f"The full df has been loaded and contain {finaldf.shape[0]} rows")
print("Example of the data:")
print(finaldf.head(3))  # present 3 first rows


print("\n -----Clean the DataFrame------- ")
# Drop the rows with null, and in addition every row that returns exception will be dropped too
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


print("\n -----The data contian 3 types of Hebrew letters: ב,ל,מ ------- ")
print(" -----Creating 3 tables to Comparisons: (ב,ל),(ב,מ),(ל,מ) ------- ")
condition1 = finaldf['category'] == 1.0  # 1 is ב
condition2 = finaldf['category'] == 2.0  # 2 is ל
condition3 = finaldf['category'] == 3.0  # 3 is מ
data23 = finaldf[~condition1]
data13 = finaldf[~condition2]
data12 = finaldf[~condition3]
data23 = data23.reset_index().drop('index', axis=1)
data13 = data13.reset_index().drop('index', axis=1)
data12 = data12.reset_index().drop('index', axis=1)

print(f"Shape of data12: {data12.shape}")
print(f"Shape of data23: {data23.shape}")
print(f"Shape of data13: {data13.shape}")