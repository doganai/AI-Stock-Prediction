import numpy
import matplotlib as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from nltk import NaiveBayesClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as arange
from sklearn.svm import SVR
from sklearn.metrics import explained_variance_score
from sklearn import linear_model

#Retrieve data, extract features and send to classifer
#BY CREATING A ANN MODEL WE CAN RETRIVE NEWS ARTICLES AND RATE AS POSITIVE, NEUTRAL OR NEGATIVE
#WITH PRICES FROM THE DATES ARTICLES WERE POSTED WE CAN TRY AND PREDICT FUTURE EVENTS
#RESEARCH SHOWS DEEP LEARNING METHODS TO BE MOST SUCCESFULL
def main():

    stock = pd.read_csv("GBPUSD.csv")

    #Organize attributes

    #REVERSE DATA SO IT STARTS FROM PAST
    stock = stock.reindex(index=stock.index[::-1])

    stock = stock[2:3950]

    frames = [stock]

    stock = pd.concat(frames)

    #GATHER DATES
    dates = np.arange(3938)

    #GATHER ALL PRICES
    prices = stock["Price"]

    #VARIANCE SCORE
    #print(explained_variance_score(days[3000:], y_rbf))

    #PREDICT PRICE FOR DAY? THE 3RD VALUE IS THE DAY
    predict_prices(dates, prices, 4000)


def predict_prices(dates, prices, x):

    #features = []

    #MOVE FEATURES FROM PANDAS TO LIST
    #for row in stock.iterrows():
     #   index, data = row
      #  features.append(data)

    #PREPROCESS DATA
    #features = preprocessing.scale(features)

    dates = np.reshape(dates, (len(dates), 1))  # converting to matrix of n X 1
    prices = np.reshape(prices, (len(prices), 1))

    predicted_price, coefficient, constant = predict(dates, prices, x)

    print(
    "\nThe stock open price for  is: $", str(predicted_price))

#GRAPH ATTRIBUTES
def predict(dates, prices, x):

    plt.xlabel("DAYS")
    plt.ylabel("PRICE")
    plt.title("FORECASTING MODEL")

    # CLASSIFIERS
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=.2)
    svr_rbf.fit(dates,prices)

    linear_mod = linear_model.LinearRegression()  # defining the linear regression model
    linear_mod.fit(dates, prices)  # fitting the data points in the model

    plt.scatter(dates, prices, color='orange', label='Data')  # plotting the initial datapoints
    plt.plot(dates, linear_mod.predict(dates), color='red', label='Linear model')  # plotting the line made by linear regression
    plt.plot(dates, svr_rbf.predict(dates), color='black',
             label='RBF')  # plotting the line made by svr_rbf

    plt.autoscale()

    plt.show()

    return linear_mod.predict(x)[0][0], linear_mod.coef_[0][0], linear_mod.intercept_[0]



main()




