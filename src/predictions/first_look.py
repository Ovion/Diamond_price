import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import BayesianRidge

from sklearn.neural_network import MLPRegressor

from sklearn.svm import SVR


data = pd.read_csv('input/clean_data')

X = data.drop(['price'], axis=1)
y = data.price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    'LR': LinearRegression(),
    'RFR': RandomForestRegressor(),
    'BR': BayesianRidge(),
    'MLP': MLPRegressor(),
    'svr': SVR()
}

for model_name, model in models.items():
    print(f'Training model: {model_name}')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)
    params = model.get_params

    with open('output/records.txt', "a+") as file:
        file.write(
            f'''Model: {model_name}\t RMSE: {rmse}\t Params: {params} \n\n'''
        )
