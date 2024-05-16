import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ----------------------------- Data Retrieval -----------------------------

# Substitute this with your valid FRED API key.
api_key = 'XXX'

# Set the identifier for S&P/Case-Shiller U.S. National Home Price Index.
series_id = 'CSUSHPINSA'

# Construct the request URL for the FRED API.
api_url = f'https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json'

# Make the request.
response = requests.get(api_url)

# Check response status.
if response.ok:
    data = response.json()['observations']
else:
    print(f"Error occurred: {response.text}")
    # Potentially raise an exception or exit the script.
    exit()

# ----------------------------- Data Cleaning -----------------------------

# Remove observations with missing values represented by '.'.
cleaned_data = list(filter(lambda obs: obs['value'] != '.', data))

# ----------------------------- Data Preparation -----------------------------

# Extract dates and values, convert all values to floating numbers.
observation_dates = [pd.to_datetime(obs['date']) for obs in cleaned_data]
price_values = [float(obs['value']) for obs in cleaned_data]

# Create a Pandas DataFrame for easier manipulation.
price_data = pd.DataFrame({'Value': price_values}, index=observation_dates)

# ----------------------------- Data Visualization -----------------------------

# Plot a figure with the specified size to make room for the labels.
plt.figure(figsize=(10, 6))

# Plotting the series data with markers.
plt.plot(price_data.index, price_data['Value'], marker='o')

# Decrease the density of x-axis labels to every five years for a clearer view.
# This sets the major locator to a 5-year interval and formats the date to display only the year.
# Modify the YearLocator interval as needed based on the span and density of your data.
plt.gca().xaxis.set_major_locator(mdates.YearLocator(5))  # Label every five years
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Display only the year

# Adding a title and labels to the plot with specified font sizes.
plt.title('S&P/Case-Shiller U.S. National Home Price Index Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Index Value', fontsize=14)

# Automatically adjust subplots to fit into the figure area.
plt.tight_layout()

# Display the plotted figure.
plt.show()

# ----------------------------- Model Training and Prediction -----------------------------

# Split the dataset into training and validation sets.
train_ratio = 0.9
train_data = price_data[:int(train_ratio * len(price_data))]
validation_data = price_data[int(train_ratio * len(price_data)):]

# Initialize the Linear Regression model.
model = LinearRegression()

# Use numeric index as feature for training and predictions.
X_train = np.arange(len(train_data)).reshape(-1, 1)
y_train = train_data['Value'].values.reshape(-1, 1)
model.fit(X_train, y_train)

# Make predictions on the training set and the validation set.
X_validation = np.arange(len(train_data), len(price_data)).reshape(-1, 1)
predictions_validation = model.predict(X_validation)
predictions_train = model.predict(X_train)

# ----------------------------- Model Evaluation -----------------------------

# Calculate RMSE and MAE on the validation set.
rmse_validation = mean_squared_error(validation_data, predictions_validation, squared=False)
mae_validation = mean_absolute_error(validation_data, predictions_validation)
ralative_error_validation = np.mean(np.abs(predictions_validation / validation_data - 1)).Value

# Calculate RMSE and MAE on the training set (to check for overfitting).
rmse_train = mean_squared_error(train_data, predictions_train, squared=False)
mae_train = mean_absolute_error(train_data, predictions_train)
ralative_error_train = np.mean(np.abs(predictions_train / train_data - 1)).Value

# Print out the metrics.
print(f"Root Mean Squared Error (Validation): {rmse_validation}")
print(f"Mean Absolute Error (Validation): {mae_validation}")
print(f"Mean Relative Error (Validation): {ralative_error_validation}")
print(f"Root Mean Squared Error (Train): {rmse_train}")
print(f"Mean Absolute Error (Train): {mae_train}")
print(f"Mean Relative Error (Train): {ralative_error_train}")

# ----------------------------- Results Visualization -----------------------------

# Visualize the fitted model and actual values over time.
plt.figure(figsize=(10, 6))
plt.plot(train_data.index, train_data, label='Train Data')
plt.plot(validation_data.index, validation_data, label='Validation Data')
plt.plot(validation_data.index, predictions_validation, label='Validation Predictions')
plt.plot(train_data.index, predictions_train, label='Training Predictions')

# Add legend and labels to the plot.
plt.legend()
plt.xlabel('Date')
plt.ylabel('Home Price Index Value')
plt.title('Predicting U.S. National Home Price Index with Linear Regression')

# Add grid lines for better readability.
plt.grid(True)

# Display the plot.
plt.show()
