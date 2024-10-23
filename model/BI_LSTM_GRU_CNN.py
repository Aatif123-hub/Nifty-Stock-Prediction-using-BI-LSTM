from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Conv1D, GRU

def build_model(input_shape):
    try:
      model = Sequential()
      model.add(Bidirectional(LSTM(50, return_sequences=True, input_shape=input_shape, activation='relu')))
      model.add(Bidirectional(LSTM(50, return_sequences=True, activation='relu')))
      model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
      model.add(GRU(50, return_sequences=True))
      model.add(LSTM(50))
      model.add(Dense(1))
      model.compile(loss='mean_squared_error', optimizer='adam')

    except Exception as e:
      raise Exception(f"Cannot load the model. Error:{e}")    
    return model