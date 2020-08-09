# LSTM Stock Predictor 

In this project, we are going to build and evalueate deep learning models using the Recurrent Neural Network(RNN) to predict the closing prices of Bitcoin. Cryptocurrency tends to volatile due to its speculative nature so traders usually try to take advantage of every resources including sentiment from social media and news article in setting up trading strategies. There is an index called **Crypto Fear & Greed Index** or **(FNG)** that analyzes the current sentiment of the Bitcoin market and crunches the numbers into a simple meter from 0 to 100. Zero means "Extreme Fear", while 100 means "Extreme Greed". Will this index be a good indicator for predicting future closing prices of Bitcoin? 

## Build models

We will build two models using the same **RNN-LSTM(Long Short Term Memory)**. One model will use the FNG(index) indicators to predict the closing price while the second model will use a window of closing prices to predict the nth closing price. We are to experiment with the model architecture and parameters to see which provides the best results. For the purpose of meaningful comparison, we will apply the same architecture and parameters to both models.

The basic idea is 
- using a 10 day window of FNG values and/or Bitcoin closing prices to predict the 11th day's closing price.
- The period of total data is from Feb 2018 to Jul 2019, or 1.5 years.
- we may change the architecture and parameters to get the best result. In more detail, we are to adjust days of window, number of input units(nodes), and batch size.
- For the architecture, we are to add two hidden layers and dropout layers at every layer before the output layer. 
- Using "adam" optimizer and "mean_squared_error" loss function for compliling.
- Epochs to be 20 times.

