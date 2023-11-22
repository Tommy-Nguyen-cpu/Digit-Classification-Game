from sklearn.svm import SVC
from joblib import dump, load
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

def TrainSVC():
    clf = SVC()
    (x_train,y_train), (x_test, y_test) = mnist.load_data()
    num_pixels = x_train.shape[1] * x_train.shape[2]
    x_train = x_train.reshape((x_train.shape[0], num_pixels)).astype('float32')
    x_test = x_test.reshape((x_test.shape[0], num_pixels)).astype('float32')

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print(accuracy_score(y_test, y_pred))

    SaveClf(clf)

def SaveClf(clf):
    dump(clf, 'clfSVC2.joblib')