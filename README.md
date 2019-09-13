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
    layers.Dense(1, activation='sigmoid'),
  ])
 

![alt text](https://github.com/JeremyCollinsMPI/Flight-delays/blob/master/dag1.png)

additionally, a feature cross between week and arrival is used.  this is to model e.g. weather events.  also year and airline, to model the inefficiency of airlines which may have changed over time (e.g. an airline has improved).

results:

with 10% of dataset as training, there is a loss of 0.0065 on the training data after 500 epochs.  it has reached convergence because it has not improved for ~350 epochs.  it has a loss of 0.08 on the test data.  


tried learning rate of 0.01 and 0.1; 0.01 seems to be fast, with batch sizes of 50,000.

current result:
validation loss of 0.0343, training loss of 0.0322, after ~40 epochs with 80-20 training-test split, and 80-20 training-validation split.

going to see the impact of the crossed features now.
removing these gets training loss of 0.0356 and validation loss of 0.0360, test loss of 0.03645

with two neurons in the first layer, the training loss at the end of 40 epochs is 0.0314 and test loss is 0.0329.  may continue going down.
with ten nuerons, training loss of 0.0293 and test loss of 0.0315 and looks like it is continuing to go down.

tried a different activation functon, 1 - (e ^ (-x)), which seems to converge at a training loss of 0.0361 and test loss of  0.0361 after 10 epochs. epochs.
