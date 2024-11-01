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

def preprocess_data(women_data, population_data):
    women_data['STATE/UT'] = women_data['STATE/UT'].str.upper()
    statewise = women_data.groupby(['STATE/UT', 'Year']).sum().reset_index()
    year_wise = statewise.groupby('Year')[['Rape', 'Kidnapping and Abduction', 'Dowry Deaths', 
                                           'Assault on women with intent to outrage her modesty', 
                                           'Insult to modesty of Women', 'Cruelty by Husband or his Relatives', 
                                           'Importation of Girls']].sum()
    year_wise['Total'] = year_wise[['Rape', 'Kidnapping and Abduction', 'Dowry Deaths', 
                                    'Assault on women with intent to outrage her modesty', 
                                    'Insult to modesty of Women', 'Cruelty by Husband or his Relatives', 
                                    'Importation of Girls']].sum(axis=1)
    merged_df = pd.merge(year_wise, population_data[['Year', 'Population', 'Literacy Rate']], on='Year', how='inner')
    merged_df['Crime_Rate'] = merged_df['Total'] / merged_df['Population'] * 1_000_000
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
        plt.scatter(y_test, predictions, label=f'{model_name}', alpha=0.6)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted for Different Models')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Diagonal line
    plt.legend()
    plt.grid(True)
    plt.show()
    return pd.DataFrame(results)

def prepare_lstm_data(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), :])  # Features including target
        y.append(data[i + time_steps, 0])  # Target: Total
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
    women_data = load_data('../Data/women/combined_data.csv')
    population_data = load_data('../population.csv')
    
    # Preprocess data
    merged_df = preprocess_data(women_data, population_data)
    
    # Define features and target column specifically for women-related crimes prediction
    features = merged_df[['Year', 'Population', 'Literacy Rate']]
    target = merged_df['Total']
    
    # Initialize traditional models for predicting women-related crimes
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate each traditional model
    print("\nEvaluating models for Women-Related Crimes Reported:")
    all_results = train_and_evaluate(models, features, target)
    
    # Display model evaluation results
    print("\nModel Evaluation Results:")
    grouped_result = all_results.groupby('Model').mean()
    print(grouped_result)
    
    # Prepare data for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(merged_df[['Total', 'Year', 'Population', 'Literacy Rate']])
    
    time_steps = 3  # Using 3 years of data to predict the next year
    X_lstm, y_lstm = prepare_lstm_data(scaled_data, time_steps)
    X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], X_lstm.shape[2])  # Reshape for LSTM
    
    # Build and train the LSTM model
    lstm_model = build_lstm_model((X_lstm.shape[1], X_lstm.shape[2]))
    lstm_model.fit(X_lstm, y_lstm, epochs=100, batch_size=32)
    
    # Save LSTM model
    lstm_model.save('lstm_model_women.h5')
    print("LSTM model saved as 'lstm_model_women.h5'")
    
    # Evaluate LSTM model
    lstm_predictions = lstm_model.predict(X_lstm)
    lstm_mse = mean_squared_error(y_lstm, lstm_predictions)
    lstm_accuracy = r2_score(y_lstm, lstm_predictions)
    print(f"\nLSTM Model Evaluation:\nMSE: {lstm_mse}\nAccuracy (R^2 score): {lstm_accuracy}")
    
    # Future predictions for LSTM
    future_years = pd.DataFrame({
        'Year': [2024, 2025, 2026],
        'Population': [1050000, 1100000, 1150000],
        'Literacy Rate': [66, 67, 68]
    })
    
    # Include year information for predictions
    future_data_full = future_years.copy()
    future_data_full['Total'] = 0  # Adding a placeholder for the scaling to be consistent
    future_data_scaled = scaler.transform(future_data_full[['Total', 'Year', 'Population', 'Literacy Rate']])
    
    # Concatenate last time_steps from scaled data with future data
    last_data = scaled_data[-time_steps:, :]
    combined_future_data = np.concatenate([last_data, future_data_scaled])
    
    # Prepare for LSTM
    future_X_lstm, _ = prepare_lstm_data(combined_future_data, time_steps)
    future_X_lstm = future_X_lstm.reshape((future_X_lstm.shape[0], future_X_lstm.shape[1], future_X_lstm.shape[2]))
    
    # Predict future values
    future_predictions = lstm_model.predict(future_X_lstm)
    future_predictions = scaler.inverse_transform(
        np.concatenate([future_predictions, future_data_scaled[:, 1:]], axis=1)
    )
    
    # Print all results together at the end
    print("\nFinal Model Evaluation Results:")
    print(grouped_result)
    print(f"\nLSTM Model Evaluation:\nMSE: {lstm_mse}\nAccuracy (R^2 score): {lstm_accuracy}")
    print("\nFuture Predictions using LSTM:")
    for year, prediction in zip(future_years['Year'], future_predictions):
        print(f"Predicted Women-Related Crimes for {year}: {prediction[0]}")

if __name__ == "__main__":
    main()
