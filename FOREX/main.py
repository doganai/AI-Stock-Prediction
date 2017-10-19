import numpy
import matplotlib as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from nltk import NaiveBayesClassifier
from yahoo_finance import Currency

#Retrieve data, extract features and send to classifer
#BY CREATING A ANN MODEL WE CAN RETRIVE NEWS ARTICLES AND RATE AS POSITIVE, NEUTRAL OR NEGATIVE
#WITH PRICES FROM THE DATES ARTICLES WERE POSTED WE CAN TRY AND PREDICT FUTURE EVENTS
#RESEARCH SHOWS DEEP LEARNING METHODS TO BE MOST SUCCESFULL
def main():

    stock = pd.read_csv("GBPUSD.csv")

    #Organize attributes
    frames = [stock[['Date','Price','Open']]]

    stock = pd.concat(frames)

    #CREATE TRAINING SET
    training = stock[1000:3938]

    #CREATE TESTING SET
    testing = stock[:1000]



main()




