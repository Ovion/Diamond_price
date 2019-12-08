import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import AdaBoostClassifier

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import BayesianRidge

from sklearn.neural_network import MLPRegressor

from sklearn.svm import SVR

from sklearn.multioutput import MultiOutputRegressor


data = pd.read_csv('input/clean_data_num.csv')

X = data.drop(['price'], axis=1)
y = data.price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    # 'LR': LinearRegression(),
    'RFR': RandomForestRegressor(n_estimators=100),
    # 'RFC': RandomForestClassifier(),
    'MOR': MultiOutputRegressor(RandomForestRegressor(n_estimators=100)),
    # 'DTR': DecisionTreeRegressor(),
    # 'KNR': KNeighborsRegressor(),
    # 'KNC': KNeighborsClassifier(),
    # 'GBR': GradientBoostingRegressor(),
    # 'GBC': GradientBoostingClassifier(),
    # 'ABR': AdaBoostRegressor(),
    # 'ABC': AdaBoostClassifier(),
    # 'GPR': GaussianProcessRegressor(),
    # 'GPC': GaussianProcessClassifier(),
    # 'MNB': MultinomialNB(),
    # 'GNB': GaussianNB(),
    # 'BR': BayesianRidge(),
    # 'MLP': MLPRegressor(),
    # 'svr': SVR()
}

for model_name, model in models.items():
    print(f'Training model: {model_name}')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)
    params = model.get_params

    print('Saving params in records.txt ...')
    with open('output/records.txt', "a+") as file:
        file.write(
            f'''Model: {model_name}\t RMSE: {rmse}\t Params: {params} \n\n'''
        )
