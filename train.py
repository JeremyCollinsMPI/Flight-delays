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

file = 'flight_delays_data.csv'
df = pd.read_csv(file, header = 0, sep = ',', na_filter=False)
df.replace({'is_claim':{800:1}})
def extract_weekday(string):
  return datetime.strptime(string, '%Y-%m-%d').weekday()
def extract_year(string):
  return datetime.strptime(string, '%Y-%m-%d').year
df['Day'] = df.flight_date.apply(extract_weekday)
df['Year'] = df.flight_date.apply(extract_year)
dataframe = df
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

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

arrival = feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list("Arrival",vocabulary_list=pd.Series.unique(df.Arrival).tolist()))
airline = feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list("Airline",vocabulary_list=pd.Series.unique(df.Airline).tolist()))
flight_no = feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list("flight_no",vocabulary_list=pd.Series.unique(df.flight_no).tolist()))

feature_columns = []
feature_columns = feature_columns + [week, arrival, airline, flight_no]
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
batch_size = 1000
train_ds = df_to_dataset(train, shuffle=True, batch_size=batch_size, target_column='is_claim')
val_ds = df_to_dataset(val, shuffle=True, batch_size=batch_size, target_column='is_claim')
test_ds = df_to_dataset(test, shuffle=True, batch_size=batch_size, target_column='is_claim') 

def train():
  model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(1, activation='relu')
  ])

  model.compile(optimizer='adam',
                loss='mean_squared_error',,
                run_eagerly=True)
  model.fit(train_ds,
            validation_data=val_ds,
            epochs=1000)
  loss = model.evaluate(test_ds)
  print("MSE Loss", loss)
  model.save('model.h5')

def evaluate():
  model = load_model('model.h5')
  loss = model.evaluate(test_ds)
  print("MSE Loss", loss)

train()





