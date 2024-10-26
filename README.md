# Stock Prediction for Nifty 50 Using a Hybrid BILSTM-GRU-CNN Model

## Overview
This project is a stock prediction application for the Nifty 50 index, built using a hybrid deep learning model that combines Bidirectional LSTM (BILSTM), GRU, and CNN layers. The application utilizes Streamlit for the web interface, allowing users to load datasets, adjust model parameters, and visualize the prediction results.

## Directory Structure
```
stock_prediction_app/
├── main.py
├── data/
│   └── Nifty 50 (2014-2024).csv
|   └── Nifty 50 (2019-2024).csv
├── model/
│   └── BILSTM_GRU_CNN.py
├── utils/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── plotting.py
├── requirements.txt
└── README.md
```

## How to Run
To run this application, you need to have Python and the required libraries installed. Follow the steps below to set up the environment and run the Streamlit app:

### Step 1: Install the dependencies
   ```bash
   pip install -r requirements.txt
   ```

### Step 2: Run the Application
Run the Streamlit app using the following command:
```bash
streamlit run main.py
```

## Application Features
- **Dataset Selection**: Choose between two datasets (2014-2024 or 2019-2024).
- **Model Parameters**: Adjust the number of epochs, batch size, and time steps dynamically through the Streamlit interface.
- **Model Training**: A hybrid model (BILSTM, GRU, CNN) is trained on the selected dataset.
- **Error Metrics**: Displays Mean Absolute Error (MAE) value for model evaluation.
- **Visualizations**:
  - Initial line graph of the dataset.
  - Actual vs. predicted stock prices.
  - Residuals between actual and predicted values.

## Files and Modules
1. **main.py**: The main Streamlit app script that orchestrates data loading, model training, and visualization.

2. **data_preprocessing.py** (in `utils/`): Contains functions for data loading, preprocessing, scaling, splitting, and creating sequences.

3. **lstm_model.py** (in `models/`): Defines the hybrid BILSTM-GRU-CNN model.

4. **model_training.py** (in `utils/`): Contains functions for training the model and calculating error metrics.

5. **plotting.py** (in `utils/`): Provides functions to generate various plots, including line graphs, residuals, and predictions.

## Requirements
The `requirements.txt` file contains all the necessary dependencies for the project. Some of the key libraries used are:
- **Streamlit**: For building the web interface.
- **TensorFlow**: For constructing and training the deep learning model.
- **Scikit-Learn**: For data preprocessing and calculating error metrics.
- **Matplotlib**: For generating visualizations.

## Usage
- After launching the app, select the dataset and specify model parameters such as epochs, batch size, and time steps.
- Click the **Submit** button to train the model and generate predictions.
- The app will display the initial line graph, prediction plot, residual plot, and MAE value.

## Example Output
- **Initial Line Graph**: Shows the historical closing price of the selected dataset.
- **Actual vs. Predicted**: Displays the model's predictions against actual data.
- **Residual Plot**: Highlights the difference between actual and predicted values.

## Notes
- Ensure that the `data/` directory contains the necessary CSV files for running the predictions.
- Training time may vary based on the selected number of epochs and batch size, as well as the computational resources available.

## Application Demo

- You can access the Application Demo which is hosted on Streamlit
- https://nifty-stock-prediction-using-bilstm-gru-cnn-ex4ezhaitgkfxggrxk.streamlit.app/

