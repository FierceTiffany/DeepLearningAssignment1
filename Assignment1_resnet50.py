import pandas as pd 
import numpy as np 
import cv2

train = pd.read_csv("train.txt", sep=" ", header=None)
test = pd.read_csv("test.txt", sep=" ", header = None)

train = train.rename(columns = {0:"image", 1:"class"})
test = test.rename(columns = {0:"image", 1:"class"})

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

train_img = []
for i in range(len(train["image"])):
    img = image.load_img(train["image"][i])
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = preprocess_input(img)
    train_img.append(img)
train_img = np.vstack(train_img)

test_img = []
for i in range(len(test["image"])):
    img = image.load_img(test["image"][i])
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = preprocess_input(img)
    test_img.append(img)
test_img = np.vstack(test_img)

train_class = np.array(train["class"])
test_class = np.array(test["class"])

resnet50 = ResNet50(weights='imagenet', include_top=False, pooling='avg')

train_features = resnet50.predict(train_img)
test_features = resnet50.predict(test_img)

# KNN 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(train_features, train_class)
y_knn_pred = knn.predict(test_features)

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

accuracy = accuracy_score(test_class, y_knn_pred)
f1 = f1_score(test_class, y_knn_pred, average = "macro")
print("KNN: Accuracy - ", accuracy, "F1 Score - ", f1)

# SVM 
from sklearn.svm import SVC
svm = SVC()
svm.fit(train_features, train_class)
y_svm_pred = svm.predict(test_features)

accuracy = accuracy_score(test_class, y_svm_pred)
f1 = f1_score(test_class, y_svm_pred, average = "macro")
print("SVM: Accuracy - ", accuracy, "F1 Score - ", f1)

# Catboost 
from catboost import CatBoostClassifier
catboost = CatBoostClassifier()
catboost.fit(train_features, train_class)
y_catboost_pred = catboost.predict(test_features)

accuracy = accuracy_score(test_class, y_catboost_pred)
f1 = f1_score(test_class, y_catboost_pred, average = "macro")
print("CatBoost: Accuracy - ", accuracy, "F1 Score - ", f1)

# Tree 
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(criterion="entropy", random_state=42)
tree.fit(train_features, train_class)
y_tree_pred=tree.predict(test_features)

accuracy = accuracy_score(test_class, y_tree_pred)
f1 = f1_score(test_class, y_tree_pred, average = "macro")
print("Tree: Accuracy - ", accuracy, "F1 Score - ", f1)