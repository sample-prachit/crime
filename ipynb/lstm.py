import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def prepare_lstm_data(data, time_steps=1):
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), :])  # Features including target
    return np.array(X)

def main():
    # Load the saved LSTM model
    lstm_model = load_model('lstm_model_murder.h5')
    # print("LSTM model loaded from 'lstm_model.h5'")
    
    # Load the previous scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Load and preprocess data
    rape_data = load_data('../Data/Violent/20_Victims_of_rape.csv')
    population_data = load_data('../population.csv')
    merged_df = preprocess_data(rape_data, population_data)
    
    # Fit the scaler with the original data columns
    scaled_data = scaler.fit_transform(merged_df[['Rape_Cases_Reported', 'Year', 'Population', 'Literacy Rate']])
    
    # Future predictions data
    future_years = pd.DataFrame({
        'Year': [2011, 2012, 2016],
        'Population': [1340935791, 1380935791, 1400935791],
        'Literacy Rate': [77, 67, 85]
    })
    
    # Include year information for predictions
    future_data_full = future_years.copy()
    future_data_full['Rape_Cases_Reported'] = 0  # Adding a placeholder for the scaling to be consistent
    future_data_scaled = scaler.transform(future_data_full[['Rape_Cases_Reported', 'Year', 'Population', 'Literacy Rate']])
    
    # Concatenate last time_steps from scaled data with future data
    time_steps = 3  # Using 3 years of data to predict the next year
    last_data = scaled_data[-time_steps:, :]
    combined_future_data = np.concatenate([last_data, future_data_scaled])
    
    # Prepare for LSTM
    future_X_lstm = prepare_lstm_data(combined_future_data, time_steps)
    future_X_lstm = future_X_lstm.reshape(future_X_lstm.shape[0], future_X_lstm.shape[1], future_X_lstm.shape[2])
    
    # Predict future values
    future_predictions = lstm_model.predict(future_X_lstm)
    future_predictions = scaler.inverse_transform(
        np.concatenate([future_predictions, future_data_scaled[:, 1:]], axis=1)
    )
    
    print("\nFuture Predictions using LSTM:")
    for year, prediction in zip(future_years['Year'], future_predictions):
        print(f"Predicted Rape Cases Reported for {year}: {prediction[0]}")

def preprocess_data(rape_data, population_data):
    rape_data['Area_Name'] = rape_data['Area_Name'].str.upper()
    statewise = rape_data.groupby(['Area_Name', 'Year']).sum().reset_index()
    year_wise = statewise.groupby('Year').sum().reset_index()
    merged_df = pd.merge(year_wise, population_data[['Year', 'Population', 'Literacy Rate']], on='Year', how='inner')
    merged_df['Crime_Rate'] = merged_df['Rape_Cases_Reported'] / merged_df['Population'] * 1_000_000
    return merged_df

if __name__ == "__main__":
    main()
