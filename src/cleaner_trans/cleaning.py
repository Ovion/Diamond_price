
import pandas as pd
from sklearn.preprocessing import StandardScaler


def magic_size(value):
    lst_magic = [0.5, 0.75, 1.0, 1.5, 1.75, 2.0, 2.5, 2.75,
                 3.0, 3.5, 3.75, 4.0, 4.5, 4.75, 5.0, 5.5, 5.75, 6.0]
    if value in lst_magic:
        return 1
    else:
        return 0


def apply_magic_clean(df):
    df.drop(['depth', 'table', 'x', 'y', 'z'], axis=1, inplace=True)
    df['magic'] = df.carat.apply(lambda x: magic_size(x))
    df_dummy = pd.get_dummies(
        data=df, columns=['cut', 'color', 'clarity'], sparse=True)
    return df_dummy


def standarization(data, test):
    lst_col = list(data.columns)
    lst_col.remove('price')

    ss = StandardScaler()
    ss.fit(data[lst_col])

    data[lst_col] = ss.transform(data[lst_col])
    test[lst_col] = ss.transform(test[lst_col])
    return data, test


if __name__ == '__main__':
    data = pd.read_csv("input/data.csv")
    test = pd.read_csv("input/test.csv")

    print('Cleaning data...')
    data_clean = apply_magic_clean(data)
    test_clean = apply_magic_clean(test)

    print('Transforming...')
    data_t, test_t = standarization(data_clean, test_clean)

    print('Saving data...')
    data_t.to_csv("input/clean_data.csv", index=False)
    test_t.to_csv("input/clean_test.csv", index=False)
    print('Done')
