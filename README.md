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

python main.py --mode evaluate

### Explanation

This repository is for predicting flight delay time, in an exercise in this link (https://t1.me/docs/assessments).  The dataset contains information on departure and arrival airports, time of the flight, airline, flight number, date of the flight, and how many hours the flight was delayed.  Flights delayed beyond 3 hours or cancelled then pay customers a fixed amount of HKD $800.  The task is to predict claim amounts for flights ($0 or $800).

This repository treats the problem as a logistic regression, coding cancelled/delayed more than three hours as 1 and anything else as 0.  The aim is to minimise the absolute error and the mean squared error.  

The model uses the following variables as one-hot vectors in the input:

Week of the year (1-52), Year (2013-2016), Hour of the day (0-23), Airline, Flight number, Destination ('Arrival')

along with feature crosses for Arrival x Week and Year x Airline.  Arrival x Week is to model seasonal events that happen in the destination, such as typhoons in places such as the Philippines; or particularly busy times of year which may cause delays in certain airports.  Year x Airline was chosen because the efficiency of certain airlines may have changed over time (e.g. an airline that had many delays one year may have had fewer the following year).

The following model in keras to predict the output of 0 or 1:

  model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(100, activation = 'sigmoid',
    layers.Dense(1, activation='sigmoid')
  ])
 
The following diagram illustrates the causal connections that are assumed in this model.  It is assumed that all of the above input variables affect the probability of a delay, and so have to be estimated jointly.  There are also causal connections between these variables (e.g. destination airport of a flight predicts the airline), and these are also included.

![alt text](https://github.com/JeremyCollinsMPI/Flight-delays/blob/master/dag1.png)

Initially, a model with just one layer predicting the output was used (i.e. model = tf.keras.Sequential([feature_layer, layers.Dense(1, activation='sigmoid')]), equvalent to a logistic regression.  An additional layer was added because it was found to reduce the test loss.

The code uses the following tensorflow tutorial template for preparing the feature columns (https://www.tensorflow.org/beta/tutorials/keras/feature_columns).

### Results

80% of the data was used for training, which was then split 80%-20% for training and validation.  The mean squared error was chosen as the loss function to minimise.  Since the output was 0 or 1, this should be multiplied by 800 to get the mean squared error for the amount in HKD of the claim: the best test data mean squared error loss is therefore (800^2) * 0.0249 = $15936.  the best mean absolute error is 0.0452, and so the best mean absolute error for the claims is 800 * 0.0452 = $36.16.

With the logistic regression, the mean squared error goes down to 0.034 on the test data.  With a fully connected layer of 100 neurons, the mean squared error on the test data is 0.0295.    

latest version with 100 neurons was trained for 49 epochs, and reached 0.0249 mean squared error. mean absolute error is 0.0452. 
when mean absolute error is reduced, it goes down to 0.0442.  the mean squared error is then 0.044.

### Further work

I could have used other crossed features, such as Hour of the day x Airline and so on.  These all may reduce the error.

I could have used other data, such as weather data for the arrival airport.  Reasons I did not: i) this is already modelled to some extent by week x arrival (seasonal effects); (ii) extreme weather events only account for 4% of flight delays (https://www.bts.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations); (iii) there is a methodological reason for not including weather data, which is that it is unknown in advance (so weather data would be useless for predicting future flight delays).  One could use past weather forecasts as data (e.g. a few days ahead), and this would be more useful.

I experimented with other activation functions besides sigmoid, including relu and a custom activation function 1 - (e ^ (-x)), neither of which improved accuracy or speed of training.

The model predicted the output of 0 or 1 directly, without predicting the amount of time of the delay.  There may be some improvement if the time is predicted directly.


