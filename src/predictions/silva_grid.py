import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('input/clean_data.csv')

X = data.drop(['price'], axis=1)
y = data.price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    'RFR_1000': RandomForestRegressor(n_estimators=1000),
    'RFR_1000_j1': RandomForestRegressor(n_estimators=1000, n_jobs=1),
    'RFR_1000_vj1': RandomForestRegressor(n_estimators=1000, verbose=1, n_jobs=1),
    'RFR_2000': RandomForestRegressor(n_estimators=2000)
}

for model_name, model in models.items():
    print(f'Training model: {model_name}')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)
    params = model.get_params

    print('Saving params in records_RFR.txt ...')
    with open('output/records_RFR.txt', "a+") as file:
        file.write(
            f'''Model: {model_name}\t RMSE: {rmse}\t Params: {params} \n\n'''
        )
