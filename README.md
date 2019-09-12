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

This repository is for predicting flight delay time in a dataset provided in this link (https://t1.me/docs/assessments).
The model uses the following variables as one-hot vectors:

Week of the year (1-52), Year (2013-2016), Hour of the day (0-23), Airline, Flight number, Destination

and the following model to predict whether a flight will be delayed by more than three hours (or cancelled):

  model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(1, activation='exponential'),
    layers.ReLU(max_value=1, negative_slope=0.0, threshold=0.0),
    layers.Lambda(lambda x:(1-x))
  ])

The activation function used is 1-exp(x).  The reasoning behind this is as follows.  Say that there probability p1 of a delay for a particular week of the year, and probability p2 of a a delay for a particular destination.  If these two probabilities are independent, then a delay can happen in this model either because of a causal factor associated with that week (e.g. it's Christmas so it is busy) or because of something associated with a particular destination (the destination airport has a problem).  For the flight to avoid the delay it has to avoid both causal factors, which it can do with probability (1 - p1) * (1 - p2), making the probability of a delay 1 - ((1 - p1) * (1 - p2)).  

This model can also take dependent probabilities by assuming that there are other causal variables dependent on e.g. both the week and the destination, such as typhoon season in Manila.  To avoid delay, the flight has to avoid the causal factor associated with that week generally (regardless of destination), the causal factor associated with that destination (regardless of time of year), and the causal factor associated with an event at the destination for that particular week (e.g. typhoon season in Manila).  

This calculation of probabilities is equivalent to 1 - exp(ln(p1) + ln(p2)), which means that the appropriate activation function for this model is 1 - exp(x).  

![alt text](https://github.com/JeremyCollinsMPI/Flight-delays/blob/master/dag1.png)

additionally, a feature cross between week and arrival is used.  this is to model e.g. weather events.  also year and airline, to model the inefficiency of airlines which may have changed over times (e.g. an airline has improved).
