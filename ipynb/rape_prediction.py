import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(rape_data, population_data):
    rape_data['Area_Name'] = rape_data['Area_Name'].str.upper()
    statewise = rape_data.groupby(['Area_Name', 'Year']).sum().reset_index()
    year_wise = statewise.groupby('Year').sum().reset_index()
    merged_df = pd.merge(year_wise, population_data[['Year', 'Population', 'Literacy Rate']], on='Year', how='inner')
    merged_df['Crime_Rate'] = merged_df['Rape_Cases_Reported'] / merged_df['Population'] * 1_000_000
    return merged_df

def train_and_evaluate(models, features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    results = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        accuracy = model.score(X_test, y_test)
        results.append({'Model': model_name, 'MSE': mse, 'Accuracy': accuracy})
        # Plotting actual vs predicted
        plt.scatter(y_test, predictions, label=f'{model_name}', alpha=0.6)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name} Predictions for Rape Cases Reported (Accuracy: {accuracy:.2f})')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
        plt.legend()
        plt.grid(True)
        plt.show()
    return pd.DataFrame(results)

def prepare_lstm_data(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), :])  # Features including target
        y.append(data[i + time_steps, 0])  # Target: Rape_Cases_Reported
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))  # Output layer for regression
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    # Load data
    rape_data = load_data('../Data/Violent/20_Victims_of_rape.csv')
    population_data = load_data('../population.csv')
    
    # Preprocess data
    merged_df = preprocess_data(rape_data, population_data)
    
    # Define features and target column specifically for rape case prediction
    features = merged_df[['Year', 'Population', 'Literacy Rate']]
    target = merged_df['Rape_Cases_Reported']
    
    # Initialize traditional models for predicting rape cases
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate each traditional model
    all_results = pd.DataFrame()
    print("\nEvaluating models for Rape Cases Reported:")
    results = train_and_evaluate(models, features, target)
    all_results = pd.concat([all_results, results], ignore_index=True)
    
    # Display model evaluation results
    print("\nModel Evaluation Results:")
    grouped_result = all_results.groupby('Model').mean()
    print(grouped_result)
    
    # Prepare data for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(merged_df[['Rape_Cases_Reported', 'Year', 'Population', 'Literacy Rate']])
    
    time_steps = 3  # Using 3 years of data to predict the next year
    X_lstm, y_lstm = prepare_lstm_data(scaled_data, time_steps)
    X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], X_lstm.shape[2])  # Reshape for LSTM
    
    # Build and train the LSTM model
    lstm_model = build_lstm_model((X_lstm.shape[1], X_lstm.shape[2]))
    lstm_model.fit(X_lstm, y_lstm, epochs=100, batch_size=32)
    
    # Save LSTM model
    lstm_model.save('lstm_model.h5')
    print("LSTM model saved as 'lstm_model.h5'")
    
    # Evaluate LSTM model
    lstm_predictions = lstm_model.predict(X_lstm)
    lstm_mse = mean_squared_error(y_lstm, lstm_predictions)
    lstm_accuracy = r2_score(y_lstm, lstm_predictions)
    print(f"\nLSTM Model Evaluation:\nMSE: {lstm_mse}\nAccuracy (R^2 score): {lstm_accuracy}")
    
    # Future predictions for LSTM
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
    last_data = scaled_data[-time_steps:, :]
    combined_future_data = np.concatenate([last_data, future_data_scaled])
    
    # Prepare for LSTM
    future_X_lstm, _ = prepare_lstm_data(combined_future_data, time_steps)
    future_X_lstm = future_X_lstm.reshape(future_X_lstm.shape[0], future_X_lstm.shape[1], future_X_lstm.shape[2])
    
    # Predict future values
    future_predictions = lstm_model.predict(future_X_lstm)
    future_predictions = scaler.inverse_transform(
        np.concatenate([future_predictions, future_data_scaled[:, 1:]], axis=1)
    )
    
    print("\nFuture Predictions using LSTM:")
    for year, prediction in zip(future_years['Year'], future_predictions):
        print(f"Predicted Rape Cases Reported for {year}: {prediction[0]}")

if __name__ == "__main__":
    main()
