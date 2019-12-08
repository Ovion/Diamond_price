import pandas as pd
from sklearn.metrics import mean_squared_error

import h2o
from h2o.automl import H2OAutoML

h2o.init()

data = h2o.import_file('input/clean_data.csv')
test = h2o.import_file('input/clean_test.csv')

X = data.columns
y = 'price'
X.remove(y)
X.remove(data['id'])

X_test = test.drop(['id'], axis=1)

aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=X, y=y, training_frame=data)

y_pred = aml.leader.predict(X_test)

submission = test['id']
submission['Price'] = y_pred
submission = submission.as_data_frame(use_pandas=True)

submission.to_csv('output/submission_aqua.csv', index=False)
