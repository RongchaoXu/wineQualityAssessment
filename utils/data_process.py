import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    filePath = "../data/winequality-red.csv"
    trainPath = "../data/red_train.csv"
    testPath = "../data/red_test.csv"
    trainSetRatio = 0.8

    df = pd.read_csv(filePath, sep=';')
    x = df.drop(columns=['quality']).copy()
    y = df['quality']

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size=trainSetRatio, test_size=1-trainSetRatio)
    X_train['quality'] = Y_train
    X_test['quality'] = Y_test
    X_train.to_csv(trainPath, index=False)
    X_test.to_csv(testPath, index=False)



