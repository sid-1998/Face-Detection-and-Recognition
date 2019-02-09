import os
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def load_images(path):
    #path = ./dataset/
    x_, y_ = [], []
    labels = os.listdir(path)
    for label in labels:
        images = os.listdir(path + label)
        for img in images:
            im = cv2.imread(path+label+"/"+img)
            im = cv2.resize(im, (100,100))
            x_.append(im)
            y_.append(label)
    return x_, y_

X, Y = load_images("./dataset/")
X = np.array(X)
Y = np.array(Y)
print(X.shape, Y.shape)

#flatten
X = X.reshape(20,-1)
print(X.shape)

X, Y = shuffle(X, Y, random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)


knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
print(knn.score(X_test, Y_test))

