from main import *

def loading_datasets(): 
  file = 'flight_delays_data.csv'
  loss_function = 'mean_squared_error'
  dataframe = create_dataframe(file)
  feature_layer = create_feature_layer(dataframe)
  train_ds, validation_ds, test_ds = create_datasets(dataframe)

def running_training():
  file = 'flight_delays_data.csv'
  loss_function = 'mean_squared_error'
  dataframe = create_dataframe(file)
  feature_layer = create_feature_layer(dataframe)
  train_ds, validation_ds, test_ds = create_datasets(dataframe)
  train(train_ds, validation_ds, test_ds, feature_layer, loss_function, epochs=1)
  
def running_evaluation():
  file = 'flight_delays_data.csv'
  loss_function = 'mean_squared_error'
  dataframe = create_dataframe(file)
  feature_layer = create_feature_layer(dataframe)
  train_ds, validation_ds, test_ds = create_datasets(dataframe)
  evaluate('models/model1/training_1/cp.ckpt', test_ds, feature_layer, loss_function) 
  
print('Loading datasets')
loading_datasets()
print('Running training')
running_training()
print('Running evaluation')
running_evaluation()
print('All tests passed')