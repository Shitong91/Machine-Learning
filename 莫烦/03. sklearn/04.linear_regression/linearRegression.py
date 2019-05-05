#from__future__import print_function
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

loaded_data = datasets.load_boston() #总共有506个数据，每个输入有13个feature
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()
model.fit(data_X, data_y)

print(model.predict(data_X[:4,:]))
print(data_y[:4])
print(model.score(data_X,data_y))
X,y=datasets.make_regression(n_samples=100,n_features=1, n_targets=1,noise=
10)
plt.scatter(X,y)
plt.show()
