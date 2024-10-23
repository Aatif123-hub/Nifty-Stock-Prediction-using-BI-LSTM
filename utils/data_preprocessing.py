import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_and_process(file_path, drop_column):
    try:
      data = pd.read_csv(file_path)
      data = data.drop(columns=[drop_column],axis=1)
      data['Date'] = pd.to_datetime(data['Date'])
    
    except Exception as e:
       raise Exception(f"Cannot process the data. Error:{e}")
    return data

def scale_data(data,feature='Close'):
    try:
      data = data.reset_index()[feature]
      scaler = MinMaxScaler(feature_range=(0, 1))
      data = scaler.fit_transform(np.array(data).reshape(-1, 1))
    
    except Exception as e:
       raise Exception(f"Cannot scale the data. Error:{e}")
    return data, scaler

def split_data(data, train_ratio=0.65):
    try: 
      training_size = int(len(data) * train_ratio)
      train_data = data[:training_size]
      test_data = data[training_size:]

    except Exception as e:
       raise Exception(f"Error while splittiong the data. Error:{e}")
    return train_data, test_data

def create_matrix(dataset, time_step=1):
    try:
      datax, datay = [], []
      for i in range(len(dataset) - time_step - 1):
         a = dataset[i:(i + time_step), 0]
         datax.append(a)
         datay.append(dataset[i + time_step, 0])

    except Exception as e:
       raise Exception(f"Error while creating a matrix.Error:{e}")
    return np.array(datax), np.array(datay)