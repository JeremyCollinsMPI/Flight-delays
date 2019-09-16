from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from datetime import datetime
import argparse 
from tensorflow.keras.constraints import non_neg
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--mode')
parser.add_argument('--path')
args = parser.parse_args()

file = 'flight_delays_data.csv'
df = pd.read_csv(file, header = 0, sep = ',', na_filter=False)
df = df.replace({'is_claim':{800:1}})

def extract_weekday(string):
  return datetime.strptime(string, '%Y-%m-%d').weekday()
def extract_year(string):
  return datetime.strptime(string, '%Y-%m-%d').year
df['Day'] = df.flight_date.apply(extract_weekday)
df['Year'] = df.flight_date.apply(extract_year)

def normalise_delay_time(string):
  threshold = 1.5
  if string == 'Cancelled':
    return 1
  else:
    number = float(string)
    if number > threshold:
      return 0.5
    else:
      return 0
df['normalised_delay_time'] = p=df.delay_time.apply(normalise_delay_time)
dataframe = df

def df_to_dataset(dataframe, shuffle=True, batch_size=32, target_column='target'):
  dataframe = dataframe.copy()
  labels = dataframe.pop(target_column)
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

week = feature_column.numeric_column("Week")
boundaries = []
for i in range(1,53):
  boundaries.append(i)
week = feature_column.bucketized_column(week, boundaries=boundaries)

day = feature_column.numeric_column("Day")
boundaries = []
for i in range(1,8):
  boundaries.append(i)
day = feature_column.bucketized_column(day, boundaries=boundaries)

year = feature_column.numeric_column("Year")
boundaries = []
for i in range(2013, 2017):
  boundaries.append(i)
year = feature_column.bucketized_column(year, boundaries=boundaries)

hour = feature_column.numeric_column("std_hour")
boundaries = []
for i in range(0,24):
  boundaries.append(i)
hour = feature_column.bucketized_column(hour, boundaries=boundaries)

arrival = feature_column.categorical_column_with_vocabulary_list("Arrival",vocabulary_list=pd.Series.unique(df.Arrival).tolist())

airline = feature_column.categorical_column_with_vocabulary_list("Airline",vocabulary_list=pd.Series.unique(df.Airline).tolist())

flight_no = feature_column.categorical_column_with_vocabulary_list("flight_no",vocabulary_list=pd.Series.unique(df.flight_no).tolist())

arrival_one_hot = feature_column.indicator_column(arrival)
airline_one_hot = feature_column.indicator_column(airline)
flight_no_one_hot = feature_column.indicator_column(flight_no)


arrival_length = len(pd.Series.unique(df.Arrival).tolist())
arrival_and_week = feature_column.crossed_column([arrival, week], hash_bucket_size=(arrival_length*52))
arrival_and_week = feature_column.indicator_column(arrival_and_week)

airline_length = len(pd.Series.unique(df.Airline).tolist())
year_and_airline = feature_column.crossed_column([year, airline], hash_bucket_size=(airline_length*4))
year_and_airline = feature_column.indicator_column(year_and_airline)

feature_columns = []
feature_columns = feature_columns + [week, arrival_one_hot, airline_one_hot, flight_no_one_hot, hour, arrival_and_week, year, year_and_airline]
# without crossed features:
# feature_columns = feature_columns + [week, arrival_one_hot, airline_one_hot, flight_no_one_hot, hour, year]

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

batch_size = 10000
test_data_size = 0.2

train, test = train_test_split(dataframe, test_size=test_data_size)
train, validation = train_test_split(train, test_size = 0.2)
train_ds = df_to_dataset(train, shuffle=True, batch_size=batch_size, target_column='is_claim')
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size, target_column='is_claim') 
validation_ds = df_to_dataset(validation, shuffle=False, batch_size=batch_size, target_column='is_claim') 

use_delay_time = True
if use_delay_time:
  delay_time_dataframe = dataframe
  delay_time_dataframe['is_claim'] = delay_time_dataframe['normalised_delay_time']
  delay_time_train, delay_time_test = train_test_split(delay_time_dataframe, test_size=test_data_size)
  delay_time_train, delay_time_validation = train_test_split(delay_time_train, test_size = 0.2)
  train_ds = df_to_dataset(delay_time_train, shuffle=True, batch_size=batch_size, target_column='is_claim')  
#   validation_ds = df_to_dataset(validation, shuffle=False, batch_size=batch_size, target_column='is_claim')

def train():
  model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(100, activation='sigmoid'),
    layers.Dense(1, activation='sigmoid')
  ])
  optimiser = tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
  model.compile(optimizer=optimiser,
                loss='mean_squared_error',
                run_eagerly=True)
  checkpoint_path = "training_1/cp.ckpt"
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
  model.fit(train_ds,
  epochs=60,
  validation_data=validation_ds,
  callbacks=[cp_callback])              
  loss = model.evaluate(test_ds)
  print("MSE Loss", loss)

def evaluate(checkpoint_path):
  model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(100, activation='sigmoid'),
    layers.Dense(1, activation='sigmoid')
  ])
  optimiser = tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
  model.compile(optimizer=optimiser,
                loss='mean_squared_error',
                run_eagerly=True)
  model.load_weights(checkpoint_path)
  loss = model.evaluate(test_ds)
  print("MSE Loss", loss)
  print('Example:')
  predictions = model.predict(test_ds, steps=1)
  print(predictions)

if args.mode == 'train':
  train()
if args.mode == 'evaluate':
  evaluate(args.path)




