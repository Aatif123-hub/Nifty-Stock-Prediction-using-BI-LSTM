import streamlit as st
from utils.data_preprocessing import load_and_process, scale_data, split_data, create_matrix
from model.BILSTM_GRU_CNN import build_model
from utils.model_training import train_model, calculate_errors
from utils.plotting import plot_predictions, plot_residuals,initial_LineGraph

st.title('Stock Prediction for Nifty 50 Using a hybrid BILSTM-GRU-CNN Model')

submit =False

dataset_option = st.selectbox("Select Dataset", ["Nifty 50 (2014-2024).csv", "Nifty 50 (2019-2024).csv"])
if dataset_option:
    epochs = st.number_input('Select number of epochs:', min_value=1, max_value=100, value=15, step=1)
    batch_size = st.number_input('Select batch size:', min_value=16, max_value=256, value=64, step=1)
    time_step = st.number_input('Select the timesteps',min_value=1,max_value=100,value=15,step=1)
    submit=st.button("Submit")
    if submit:
     file_path = f'data/{dataset_option}'
     data = load_and_process(file_path,'Index Name')
     data, scaler = scale_data(data)
     train_data, test_data = split_data(data)
     x_train, y_train = create_matrix(train_data, time_step)
     x_test, y_test = create_matrix(test_data, time_step)
     x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
     x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

     model = build_model((time_step, 1))
     train_predict, test_predict = train_model(model, x_train, y_train, x_test, y_test,epochs,batch_size)
     train_predict = scaler.inverse_transform(train_predict)
     test_predict = scaler.inverse_transform(test_predict)

     initial_LineGraph(data)
     plot_predictions(data, train_predict, test_predict, scaler, time_step)
     plot_residuals(data, test_predict, scaler, time_step, len(train_predict))

     st.image('plots/Initial Line Graph.png')
     st.image('plots/plot_predictions.png')
     st.image('plots/plot_residuals.png')
     mae = calculate_errors(y_test,test_predict)
     st.write(f"MAE Value:{mae}")
     st.write("Training completed.")
    