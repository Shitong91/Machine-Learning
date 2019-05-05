import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

#load data
digits = load_digits()
X = digits.data
y = digits.target
print(y)
y = LabelBinarizer().fit_transform(y)
print(y[:3,:])
#Binarize labels in a one-vs-all fashion
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


