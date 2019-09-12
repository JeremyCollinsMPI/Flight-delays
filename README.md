# Flight-delays
### Set-up


Dependencies:
python 3.5
pip install pandas matplotlib tensorflow==2.0.0-rc0 SciPy numpy sklearn

### Docker image


Alternatively run this docker container, which also contains the repository:
docker run -it --rm jeremycollinsmpi/flight-data:latest

To run the script:

### Running the script



python main.py --mode train

or 

python main.py --mode evaluate

### Explanation

This repository is for predicting flight delay time in a dataset provided in this link (https://t1.me/docs/assessments).
The model uses the following variables as one-hot vectors in a layer called 
Week of the year (1-52), Year (2013-2017), Hour of the day (0-23), Airline, Flight number, Destination
and the following model to predict whether a flight will be delayed by more than three hours (or cancelled):
tf.keras.Sequential([
    feature_layer,
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='relu')
  ])


![alt text](https://github.com/JeremyCollinsMPI/Flight-delays/blob/master/dag1.png)
