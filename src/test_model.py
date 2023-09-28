import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from joblib import dump, load


parser = argparse.ArgumentParser(description ='Code to train the model')
parser.add_argument('--data', type = str, help ='data file path')
parser.add_argument('--model', type = str, help ='model file path')
parser.add_argument('--scaler', type = str, help ='standard scaler file path')
parser.add_argument('--encoder', type = str, help ='encoder file path')

args = parser.parse_args()
file = args.data # file path


columns = ["Commodity", "Year", "arrivals_in_qtl", "modal_price"]
monthData = pd.read_csv(file)
monthData = monthData[columns] # Selecting the required columns

# Encoding the columns
with open(args.encoder, "rb") as f:
    enc = load(f)
enc_data = pd.DataFrame(enc.fit_transform(monthData[["Commodity"]]).toarray())

new_data = pd.concat([enc_data, monthData], axis=1)
new_data.head()

columns_to_keep = list(new_data.columns)
# columns_to_keep.remove("APMC")
columns_to_keep.remove("Commodity")
# columns_to_keep.remove("Month")

new_data = new_data[columns_to_keep]

train_len = int(new_data.shape[0] * 0.8)

# Shuffling dataset
new_data = new_data.sample(frac=1)
# train = new_data.iloc[:train_len, :]
test = new_data.iloc[train_len:, :]

# print("Train set shape:", train.shape)
print("Test set shape:", test.shape)

# train = train.to_numpy()
test = test.to_numpy()

scaler = load(args.scaler)
# train = scaler.fit_transform(train)
test = scaler.transform(test)

# X_train, y_train = train[:, :-1], train[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]


model = load(args.model)
# model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("RMSE Score of the Linear Regression Model: ", mean_squared_error(y_test, y_pred))




