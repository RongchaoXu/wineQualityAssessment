from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from dataset import get_dataset
from utils.analysis import get_distribution
from imblearn.under_sampling import *
from imblearn.over_sampling import *
from imblearn.combine import *
from imblearn.ensemble import *


def logistic(solver='newton-cg', max_iter=10000):
    return LogisticRegression(solver=solver, max_iter=max_iter)


def imbalanced_learning(x, y):
    cc = TomekLinks()
    x_res, y_res = cc.fit_resample(x, y)
    print(y.shape, y_res.shape)
    print(get_distribution(y_res))
    return x_res, y_res


def train_model(x_train, y_train, x_test, y_test):
    # x_train, y_train = imbalanced_learning(x_train, y_train)

    model = logistic()
    # model = KNeighborsClassifier(n_neighbors=10)
    # model = SVC(kernel='poly')
    model.fit(x_train, y_train)
    pred_labels = model.predict(x_test)
    acc = accuracy_score(y_test, pred_labels)
    f1 = f1_score(y_test, pred_labels, average=None)
    print(acc, f1)


if __name__ == '__main__':
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    x_train, y_train, x_test, y_test = get_dataset(train_path, test_path)
    train_model(x_train, y_train, x_test, y_test)