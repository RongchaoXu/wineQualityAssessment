import pandas as pd


def get_dataset(train_path, test_path):
    train = pd.read_csv(train_path, sep=',')
    test = pd.read_csv(test_path, sep=',')
    x_train = train.drop(columns=['quality']).copy()
    y_train = train['quality'].to_frame()
    x_test = test.drop(columns=['quality']).copy()
    y_test = test['quality'].to_frame()
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    result = get_dataset('data/train.csv', 'data/test.csv')
    # print(result[0].shape, result[1].shape, result[2].shape, result[3].shape)
    print(result[1])