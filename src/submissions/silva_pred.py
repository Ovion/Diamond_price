import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('input/clean_data_num.csv')
test = pd.read_csv('input/clean_test_num.csv')

X_train = data.drop(['price'], axis=1)
y_train = data.price
X_test = test.drop(['id'], axis=1)

print('Training...')
rfr = RandomForestRegressor(n_estimators=500, n_jobs=-1)
rfr.fit(X_train, y_train)
print('Doing a prediction...')
y_pred = rfr.predict(X_test)

print('Saving...')
submit = pd.DataFrame({
    'id': test['id'],
    'price': y_pred
})

submit.price = submit.price.apply(lambda x: round(x, 0))
submit.price = submit.price.apply(lambda x: int(x))

submit.to_csv('output/submit_silva_woM_500.csv', index=False)
print('Done')
