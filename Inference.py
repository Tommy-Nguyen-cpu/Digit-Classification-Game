from sklearn.svm import SVC
from joblib import load
import cv2

def Predict(data):
    clf = load('clfSVC2.joblib')
    return clf.predict(ConvertData(data))

def ConvertData(data):
    grayScale = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
    shrunk_data = cv2.resize(grayScale, (28,28))
    return (shrunk_data.reshape(1,784))