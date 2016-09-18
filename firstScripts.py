from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
x = iris.data[:,[2,3]]
y = iris.target

#print(len(y))


from sklearn.cross_validation import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=0)

print(len(yTrain))
print(len(yTest))


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(xTrain)

xTrainStd = sc.transform(xTrain)
xTestStd = sc.transform(xTest)


print(xTrainStd)


from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(xTrainStd, yTrain)





