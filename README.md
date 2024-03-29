# Flight-delays
### Set-up


Dependencies:
python 3.5
pip install pandas matplotlib tensorflow==2.0.0-rc0 SciPy numpy sklearn

Data is in this link: https://drive.google.com/a/terminal1.co/file/d/1AkEc76q6NbqEojk3BQJEfbx-RIigDCve/view?usp=sharing

### Docker image


Alternatively run this docker container, which comes with the repository and data included:

docker run -it --rm jeremycollinsmpi/flight-data:latest /bin/bash


### Running the script



python main.py --mode train

or 

python main.py --mode evaluate --path models/model1/training_1/cp.ckpt

### Explanation

This repository is for predicting flight delay time, in an exercise in this link (https://t1.me/docs/assessments).  The dataset contains information on departure and arrival airports, time of the flight, airline, flight number, date of the flight, and how many hours the flight was delayed.  Flights delayed beyond 3 hours or cancelled then pay customers a fixed amount of HKD $800.  The task is to predict claim amounts for flights ($0 or $800).

This repository treats the problem as a logistic regression, coding cancelled/delayed more than three hours as 1 and anything else as 0.  The aim is to minimise the absolute error and the mean squared error.  

The model uses the following variables as one-hot vectors in the input:

Week of the year (1-52), Year (2013-2016), Hour of the day (0-23), Airline, Flight number, Destination ('Arrival')

along with feature crosses for Arrival x Week and Year x Airline.  Arrival x Week is to model seasonal events that happen in the destination, such as typhoons in places such as the Philippines; or particularly busy times of year which may cause delays in certain airports.  Year x Airline was chosen because the efficiency of certain airlines may have changed over time (e.g. an airline that had many delays one year may have had fewer the following year).

The following model in keras to predict the output of 0 (no claim) or 1 (claim):

  model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(100, activation = 'sigmoid',
    layers.Dense(1, activation='sigmoid')
  ])
 
The following diagram illustrates the causal connections that are assumed in this model.  It is assumed that all of the above input variables affect the probability of a delay, and so have to be estimated jointly.  There are also causal connections between these variables (e.g. destination airport of a flight predicts the airline), and these are also included.  In the diagram, 'Airline_problems' and 'Seasonal_events' are variable names for the feature crosses included in the model, Airline x Year and Arrival x Week.  

![alt text](https://github.com/JeremyCollinsMPI/Flight-delays/blob/master/dag1.png)

Initially, a model with just one layer predicting the output was used (i.e. model = tf.keras.Sequential([feature_layer, layers.Dense(1, activation='sigmoid')]), equvalent to a logistic regression.  An additional layer was added because it was found to reduce the test loss (increasing the number of neurons did not subsequently decrease the test loss).  The model was trained using the Adam optimisation algorithm with a learning rate of 0.01 (other learning rates such as 0.1 did not shorten the training time).

The code uses the following tensorflow tutorial template for preparing the feature columns (https://www.tensorflow.org/beta/tutorials/keras/feature_columns).

### Results

80% of the data was used for training, which was then split 80%-20% for training and validation.  The mean squared error was chosen as the loss function to minimise.  Since the output was 0 or 1, this should be multiplied by 800^2 to get the mean squared error for the amount in HKD of the claim: the best test data mean squared error loss is therefore (800^2) * 0.0249 = $15936.  The best mean absolute error is 0.0442 (model 2), and so the best mean absolute error for the claims is 800 * 0.0442 = $35.36

With the logistic regression, the mean squared error goes down to 0.034 on the test data.  With a fully connected layer of 100 neurons, the mean squared error on the test data reached a mean squared error of 0.0249 after 49 epochs (and did not subsequently decrease, suggesting that it had converged).  The mean absolute error is 0.0452.  When the model is run with mean absolute error as the loss instead, the loss goes down to 0.0442, while the mean squared error is then 0.044.  These two models are saved under models/model1 and models/model2 respectively.

I experimented with other activation functions besides sigmoid, including relu and a custom activation function 1 - (e ^ (-x)), neither of which improved accuracy or speed of training.

The model predicted the output of 0 or 1 directly, without predicting the amount of time of the delay.  There may be some improvement if the time is predicted directly.  To test this I coded delays of more than 1.5 hours as 0.5 (0 being no claim and 1 being a claim), to test whether the model trained on this data would then be more accurate in predicting the unchanged test data.  This was run for 100 epochs, and the validation mean squared error had gone down to 0.0295, higher than the 0.0249 achieved without that change to the data.  Further work is needed to use to predict the delay time itself. 

### Further work

I could have used other crossed features, such as Hour of the day x Airline and so on.  These all may reduce the error, with more time for experimentation.  

It would be useful to investigate the weights of the layers, which could be interpreted as the relative effects of different destinations, weeks of the year and so on on the final probability of a delay.  It would also be useful to compare the results of this model with another baseline such as random forests.

The model could use other data such as weather forecasts (forecasts of extreme weather events in particular could predict delays).  To some extent this is already modelled by the feature cross Arrival x Week, which is partly in order to model seasonal changes in the probability of delay in particular destinations.  Extreme weather events account for 4% of delays (https://www.bts.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations) and so may help improve the accuracy of the model.





