
import pandas as pd
from sklearn.preprocessing import StandardScaler


def num_clarity(value):
    dict_clarity = {
        'I1': 1,
        'SI1': 2,
        'SI2': 3,
        'VS1': 4,
        'VS2': 5,
        'VVS1': 6,
        'VVS2': 7,
        'IF': 8
    }
    return dict_clarity[value]


def num_cut(value):
    dict_cut = {
        'Fair': 1,
        'Good': 2,
        'Very Good': 3,
        'Premium': 4,
        'Ideal': 5
    }
    return dict_cut[value]


def num_color(value):
    dict_color = {
        'J': 1,
        'I': 2,
        'H': 3,
        'G': 4,
        'F': 5,
        'E': 6,
        'D': 7
    }
    return dict_color[value]


def magic_size(value):
    lst_magic = [0.5, 0.75, 1.0, 1.5, 1.75, 2.0, 2.5, 2.75,
                 3.0, 3.5, 3.75, 4.0, 4.5, 4.75, 5.0, 5.5, 5.75, 6.0]
    if value in lst_magic:
        return 1
    else:
        return 0


def apply_magic_clean(df):
    df.drop(['x', 'y', 'z'], axis=1, inplace=True)
    df['magic'] = df.carat.apply(lambda x: magic_size(x))
    df.clarity = df.clarity.apply(lambda x: num_clarity(x))
    df.color = df.color.apply(lambda x: num_color(x))
    df.cut = df.cut.apply(lambda x: num_cut(x))
    return df


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
    data_t.to_csv("input/clean_data_num.csv", index=False)
    test_t.to_csv("input/clean_test_num.csv", index=False)
    print('Done')
