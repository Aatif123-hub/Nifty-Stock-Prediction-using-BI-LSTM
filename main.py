import streamlit as st
import numpy as np
from utils.data_preprocessing import load_and_process, scale_data, split_data, create_matrix
from model.BI_LSTM_GRU_CNN import build_model
from utils.model_training import train_model, calculate_errors
from utils.plotting import plot_predictions, plot_future_predictions, plot_residuals

st.title('Stock Prediction for Nifty 50 using a Hybrid BILSTM-CNN-GRU Model ')

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    data = load_and_process(uploaded_file, 'Index Name')
    data, scaler = scale_data(data)
    train_data, test_data = split_data(data)
    time_step = 15
    x_train, y_train = create_matrix(train_data, time_step)
    x_test, y_test = create_matrix(test_data, time_step)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    model = build_model((time_step, 1))
    train_predict, test_predict = train_model(model, x_train, y_train, x_test, y_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    plot_predictions(data, train_predict, test_predict, scaler, time_step)
    plot_residuals(data, test_predict, scaler, time_step, len(train_predict))


    st.image('plots/plot_predictions.png')
    st.image('plots/plot_residuals.png')
    mae = calculate_errors(y_test,test_predict)
    st.write(f"MAE Value:{mae}")
    st.write("Training completed.")
    