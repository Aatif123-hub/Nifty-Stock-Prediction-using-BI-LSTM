import matplotlib.pyplot as plt
import numpy as np


def initial_histogram(data):
    plt.plot(data)
    plt.title(f"Closing price of Nifty 50")
    plt.savefig('plots/initial_histogram.png')
    plt.close()

def plot_boxplot(data):
    plt.figure(figsize=(12, 6))
    plt.boxplot(data, patch_artist=True)
    plt.title(f'Boxplot of Closing Prices')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.savefig('plots/boxplot_prices.png')
    plt.close()

def plot_predictions(data, train_predict, test_predict, scaler, time_step):
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
    plt.savefig('plots/plot_predictions.png')
    plt.close()


def plot_residuals(data, test_predict, scaler, time_step, train_predict_len):
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
    plt.savefig('plots/plot_residuals.png')
    plt.close()