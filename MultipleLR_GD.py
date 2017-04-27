import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

#Taking the King County housing data from Kaggle to extract 3 features for
#gradient descent

def main():
    data = pd.read_csv('kc_house_data.csv')
    data = data.as_matrix()

    #Extracting only 3 features
    X = data[:, 3:6].astype(np.float32) 
    Y = data[:, 2]

    #Normalizing the data
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0)
    np.place(std, std == 0, 1) 
    X = (X - mean)/std

    #Adding 1s for bias
    ones = np.zeros(21613).reshape(21613,1)
    ones[:] = 1
    X = np.hstack((ones, X))

    N, D = X.shape

    lr = 0.000001
    w = np.random.randn(D)/ np.sqrt(D)

    cost = []

    #Set up gradient descent
    for i in xrange(1500):
        Y_hat = X.dot(w)
        delta = (Y_hat - Y)
        w = w - lr * X.T.dot(delta) 
        mse = delta.dot(delta)/N
        cost.append(mse)
        
    plt.plot(cost)
    plt.show()

    print "The final weights are:", w


    #Measure against Sklearn
    regr = LinearRegression()
    regr.fit(X,Y)
    print "Sklearn weights are:", regr.coef_

if __name__ == '__main__':
    main()

