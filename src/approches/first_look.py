
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import BayesianRidge

from sklearn.neural_network import MLPRegressor

from sklearn.svm import SVR

from sklearn.multioutput import MultiOutputRegressor

'''
Hola... Qué mirando a ver cómo ganas la competición de Kaggle??? Solo decir que yo no la gané
Pero en archivos ocultos estoy intentando ganaros a vosotros
Muwahahahahahah

No obstante he retocado este archivo para que tengáis algo muy útil de por donde tirar
'''


data = pd.read_csv('input/clean_data_num.csv')

X = data.drop(['price'], axis=1)
y = data.price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    # 'LR': LinearRegression(n_jobs=-1),
    # 'RFR0': RandomForestRegressor(n_estimators=100, n_jobs=-1, criterion='mae'),
    'RFR1': RandomForestRegressor(n_estimators=100, n_jobs=-1),
    # 'KNR0': KNeighborsRegressor(n_jobs=-1, weights='distance'),
    # 'KNR1': KNeighborsRegressor(n_jobs=-1, weights='uniform'),
    # 'KNR2': KNeighborsRegressor(n_jobs=-1, weights='uniform', algorithm='ball_tree'),
    # 'KNR3': KNeighborsRegressor(n_jobs=-1, weights='uniform', algorithm='kd_tree'),
    # 'KNR4': KNeighborsRegressor(n_jobs=-1, weights='uniform', algorithm='brute'),
    # 'MOR': MultiOutputRegressor(RandomForestRegressor(n_estimators=100)),
    # 'DTR': DecisionTreeRegressor(),
    'GBR': GradientBoostingRegressor(validation_fraction=0.2),
    # 'ABR': AdaBoostRegressor(),
    # 'GPR': GaussianProcessRegressor(),
    # 'MNB': MultinomialNB(),
    # 'GNB': GaussianNB(),
    # 'BR': BayesianRidge(),
    # 'MLP': MLPRegressor(),
    # 'svr': SVR()
}

for model_name, model in models.items():
    print(f'Training model: {model_name}')
    # Entrenando para cada modelo del diccionario
    lst_success = []
    lst_rmse = []
    for i in range(6):
        # Este es mi propio Cross Validation
        print(f'Iteration number {i}')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        dpre = pd.DataFrame({
            'pred': y_pred,
            'GT': y_test})

        success = round(
            (dpre[dpre.pred == dpre.GT].shape[0]/len(dpre.pred))*100, 2)
        lst_success.append(success)
        # Esta es una lista donde voy guardando lo que he acertado (Acertar el precio exacto es prácticamente imposible)
        lst_rmse.append(mean_squared_error(y_test, y_pred)**0.5)
        # Esta es una lista donde voy almacenando el error cuadrático medio

    print("Saving data at 'output/records.txt'...")
    # Tengo un log de todo lo que voy haciendo
    success_mean = round(sum(lst_success)/len(lst_success), 2)
    rmse_mean = round(sum(lst_rmse)/len(lst_rmse), 2)
    params = model.get_params
    with open('output/records.txt', "a+") as file:
        file.write(
            f'''Model: {model_name}\t Success: {success_mean}%\t RMSE: {rmse_mean}%
            \n\tParams: {params} \n\n'''
        )
    print(f'Model {model_name} analyzed')
