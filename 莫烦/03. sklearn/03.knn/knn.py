import numpy as np
from sklearn import datasets  #从sklearn中导出数据集
from sklearn.model_selection import train_test_split #随机的把数据集分为train and test subsets
from sklearn.neighbors import KNeighborsClassifier# classifier implementing the K-nearest neighbors vote

iris = datasets.load_iris() #该数据集分为3classes，每个输入数据是站1×4,一共有150个数据。
iris_X =iris.data
iris_y =iris.target

#print(iris_X[:2,:])         #去前两行的数据
#print(iris_y)

X_train, X_test, y_train, y_test = train_test_split(
        iris_X, iris_y, test_size=0.3)    #150个数据的30%作为test dataset

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print(knn.predict(X_test))
print(y_test)


