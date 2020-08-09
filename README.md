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

## Finding the best model

For the **FNG model**, we started using all parameters set at one( `window = 1`, `number_units = 1`, `batch_size=1`). The result gives us a plot of predicted closing prices.    
![FNG_w1_i1_b1](https://github.com/coolwonny/LSTM-Stock-Predictor/blob/master/Images/FNG_w1_i1_b1.png)   

As shown in the plot, the performance looks not well enough. We now adjust the `number_units` or *number of inputs* to 30 while other parameters remain unchanged. Then we got this plot as below.    
![FNG_w1_i30_b1](https://github.com/coolwonny/LSTM-Stock-Predictor/blob/master/Images/FNG_w1_i30_b1.png) 
   
Again, we are not satisfied with the result. Let's increase the window = 10, number of inputs = 10 and batch size = 1. Then, we come up with this flatter plot as below.   

![FNG_w10_i10_b1](https://github.com/coolwonny/LSTM-Stock-Predictor/blob/master/Images/FNG_w10_i10_b1.png)

The plot became much flatter if we increased the number of input = 30 and batch size = 10 like the figure below.
![FNG_w10_i30_b10](https://github.com/coolwonny/LSTM-Stock-Predictor/blob/master/Images/FNG_w10_i30_b10.png)

We have repeated this kind of modification over and over again to see how the plot is changed. You may take a look at those result in **[Images](https://github.com/coolwonny/LSTM-Stock-Predictor/tree/master/Images)**. For your reference, each file name follows this rule; FNG or CL(Closing Price) denotes which model it is, and 'w'means window, 'i'means number of inputs, and 'b'means batch size. All numbers following them are parameter values used in each test. For example, if the file name is 'FNG_w1_i30_b10' meaning that it used FNG model with parameter values of window = 1, number of inputs = 30 and batch size = 10. 

One finding from the trial and error was that the model gives us not worse results as window is close to 1 and number of inputs become larger, together with the batch size. Of course, you might find a strange result when the parameter becomes large enough, so we had to find a compromising point that looks better than others. 

Honestly, it was difficult to find the best-looking plot from the FNG model. However, if we have to determine one, we would pick this `window=1, input=30, batch size = 20` one up.   
![FNG_w1_i30_b20](https://github.com/coolwonny/LSTM-Stock-Predictor/blob/master/Images/FNG_best_w1_i30_b20.png)   
   
**In contrast**, we had easy times with picking the best candidate from the **Closing Price model**. Like the previous model, we started looking at the result with 'w1_i1_b1'(I suppose you are now familiar with this notation).   
![CL_w1_i1_b1](https://github.com/coolwonny/LSTM-Stock-Predictor/blob/master/Images/CL_w1_i1_b1.png)   

As you can see the above, the plot shows relatively better performance than all of the FNG models. Now we tested 'w1_i10_b10' which returns us a plot as below.   

![CL_w1_i10_b10](https://github.com/coolwonny/LSTM-Stock-Predictor/blob/master/Images/CL_w1_i10_b10.png)

This is definitely better than before. Likewise, we have done a various trial and error only to get this best plot at 'w1_i30_b20' as below.   
![CL_w1_i30_b20](https://github.com/coolwonny/LSTM-Stock-Predictor/blob/master/Images/CL_best_w1_i30_b20.png)

This incredible prediction plot has a loss of 0.00163, or putting it another way, almost reached 98.4% accuracy rate in prediction! 

## Model Comparison

Now we come to determine which model performs best in comparison to the other. This was truly the easiest part of the project by simply taking a look at the best plot from each model.    

![FNG Best](https://github.com/coolwonny/LSTM-Stock-Predictor/blob/master/Images/FNG_best_w1_i30_b20.png)   ![CL Best](https://github.com/coolwonny/LSTM-Stock-Predictor/blob/master/Images/CL_best_w1_i30_b20.png)

#### Which model has a lower loss?    
> Closing Price model showed a lower loss at 0.00163098 than FNG model, which has the lowest loss at 0.07690528    

#### Which model tracks the actual values better over time?     
> Closing Price model does. It tracks the actual values precisely while FNG model tracks it haphazardly.    

#### Which window size works best for the model?   
> According to the tests, the smaller window size the better. We got our best model at window size equals one.    

## Conclusion

In conclusion, we found that using Closing Price as both of the feature and the output gives us a lot better performance in predicting future closing prices of Bitcoin than using FNG index as a feature. This can be attributable to the characteristic of RNN-LSTM that feeds in the output of the previous time-data(or t-1) to train the model. Therefore, it is important to use the features and target in the same series of dataset to get the best result from the RNN-LSTM model. We demonstrated it by comparing the FNG model that differentiated the source of features(FNG values) and targets(Closing prices) to the Closing Price model that synchronized the sources within the same dataset(closing prices).
     

You may refer to the Jupyter notebook files by clicking the file name below.    
[lstm_stock_predictor_fng.ipynb](https://github.com/coolwonny/LSTM-Stock-Predictor/blob/master/lstm_stock_predictor_fng.ipynb)    
[lstm_stock_predictor_closing.ipynb](https://github.com/coolwonny/LSTM-Stock-Predictor/blob/master/lstm_stock_predictor_closing.ipynb)


