import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import h2o
from h2o.automl import H2OAutoML

h2o.init()

data = pd.read_csv('input/clean_data.csv')

X = data.drop(['price'], axis=1)
y = data.price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
