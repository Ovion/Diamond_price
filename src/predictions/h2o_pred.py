import pandas as pd
from sklearn.metrics import mean_squared_error

import h2o
from h2o.automl import H2OAutoML

h2o.init(nthreads=-1, max_mem_size=10)

data = h2o.import_file('input/clean_data_num.csv')
pred = h2o.import_file('input/clean_test_num.csv')

splits = data.split_frame(ratios=[0.75], seed=1)
train = splits[0]
test = splits[1]

y = 'price'

X_test = pred.drop(['id'], axis=1)

aml = H2OAutoML(max_runtime_secs=300, seed=1, sort_metric="RMSE", nfolds=0)
aml.train(y=y, training_frame=train, validation_frame=test)

y_pred = aml.leader.predict(X_test)

submit = pred['id']
submit['price'] = y_pred
submit = submit.as_data_frame(use_pandas=True)

submit.price = submit.price.apply(lambda x: round(x, 0))
submit.price = submit.price.apply(lambda x: int(x))

submit.to_csv(
    'output/submit_aqua_0.75_int_wodt_wValidation.csv', index=False)

print('Saving params in records.txt ...')
with open('output/records_aqua.txt', "a+") as file:
    file.write(
        f'''Models:\n {aml.leaderboard} \n\n'''
    )

h2o.cluster().shutdown()
