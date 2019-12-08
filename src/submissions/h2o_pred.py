import pandas as pd
from sklearn.metrics import mean_squared_error

import h2o
from h2o.automl import H2OAutoML

h2o.init()

data = h2o.import_file('input/clean_data_num.csv')
test = h2o.import_file('input/clean_test_num.csv')

X = data.columns
y = 'price'
X.remove(y)
X.remove(data['id'])

X_test = test.drop(['id'], axis=1)

aml = H2OAutoML(max_models=40, seed=1, sort_metric="RMSE")
aml.train(x=X, y=y, training_frame=data)

y_pred = aml.leader.predict(X_test)

submit = pd.DataFrame({
    'id': test['id'],
    'price': y_pred
})

submit.price = submit.price.apply(lambda x: round(x, 0))
submit.price = submit.price.apply(lambda x: int(x))

submit.to_csv('output/submit_aqua.csv', index=False)
