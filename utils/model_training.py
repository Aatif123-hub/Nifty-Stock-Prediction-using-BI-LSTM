import math
from sklearn.metrics import mean_absolute_error,root_mean_squared_error

def train_model(model, x_train, y_train, x_test, y_test, epochs, batch_size):
    try:
      model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1)
      train_predict = model.predict(x_train)
      test_predict = model.predict(x_test)

    except Exception as e:
       raise Exception(f"Cannot train the model. Error:{e}")
    return train_predict, test_predict

def calculate_errors(y_true, y_pred):
    try:
      mae = math.sqrt(mean_absolute_error(y_true, y_pred))
    
    except Exception as e:
      raise Exception(f"Cannot calculate the error. Error:{e}")
    
    return mae
