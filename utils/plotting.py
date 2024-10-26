import matplotlib.pyplot as plt
import numpy as np


def initial_LineGraph(data):
    try:
      plt.figure(figsize=(12, 6))
      plt.plot(data)
      plt.title(f"Closing price of Nifty 50")
      plt.savefig('plots/Initial Line Graph.png')
      plt.close()
    except Exception as e:
        raise Exception(f"Cannot load Line Graph. Error:{e}")

def plot_predictions(data, train_predict, test_predict, scaler, time_step):
    try:
      predictions_plot = np.empty_like(data)
      predictions_plot[:, :] = np.nan
      train_start_index = time_step
      train_end_index = train_start_index + len(train_predict)
      predictions_plot[train_start_index:train_end_index, :] = train_predict
      test_start_index = train_end_index + time_step
      test_end_index = test_start_index + len(test_predict)
      predictions_plot[test_start_index:test_end_index, :] = test_predict

      plt.figure(figsize=(15, 8))
      plt.plot(scaler.inverse_transform(data), label='Actual Data')
      plt.plot(predictions_plot, label='Predicted Data', color='r')
      plt.xlabel('Days')
      plt.title('Actual vs Predicted')
      plt.ylabel('Stock Price')
      plt.legend()
      plt.savefig('plots/predictions.png')
      plt.close()
    except Exception as e:
       raise Exception(f"Cannot load Prediction plot. Error:{e}")


def plot_residuals(data, test_predict, scaler, time_step, train_predict_len):
    try:
     test_start_index = train_predict_len + (time_step * 2) + 1
     test_end_index = len(data) - 1
     actual_test_data = scaler.inverse_transform(data[test_start_index:test_end_index, :])
     residuals = actual_test_data - test_predict

     plt.figure(figsize=(12, 6))
     plt.plot(residuals, label='Residuals (Actual - Predicted)', color='red')
     plt.axhline(0, color='black', linestyle='--')
     plt.legend()
     plt.title("Residuals for Test Predictions")
     plt.xlabel("Days")
     plt.ylabel("Residual Value (Difference)")
     plt.savefig('plots/residuals.png')
     plt.close()
    except Exception as e:
       raise Exception(f"Cannot load the Residual plot. Error:{e}")