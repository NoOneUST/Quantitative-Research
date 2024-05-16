# Importing required packages
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
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
    data_json = response.json()
else:
    print(f"Error occurred: {response.text}")
    # Potentially raise an exception or exit the script.
    exit()

# ----------------------------- Data Cleaning -----------------------------

# Data preparation
dates = [obs['date'] for obs in data_json['observations'] if obs['value'] != '.']
values = [float(obs['value']) for obs in data_json['observations'] if obs['value'] != '.']
dates = pd.to_datetime(dates)

# Normalize the data into a specified range for ease of processing
scaler = MinMaxScaler(feature_range=(-1, 1))
values_normalized = scaler.fit_transform(np.array(values).reshape(-1, 1)).reshape(-1)

# Checking availability of GPU. It will use CUDA if available, else CPU will be used.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------- Model Define ------------------------------

# Define LSTM model
class LSTMModel(nn.Module):
    # LSTMModel class inherits from torch.nn.Module, a base class for all neural network modules
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()  # inherits properties from Module class
        self.hidden_dim = hidden_dim  # number of hidden layer dimensions
        self.num_layers = num_layers  # number of hidden layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)  # applies LSTM to input data
        self.linear = nn.Linear(hidden_dim, output_dim)  # applies linear transformation to incoming data

    # Defines the computation performed in every call to the model
    def forward(self, x):
        # initial hidden state for each element in the batch
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        # initial cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))  # passes the input and hidden states into the LSTM
        out = self.linear(out[:, -1, :])  # passes the last output tensor into the linear layer
        return out


# Function to create sequences
def create_sequences(data, seq_length):
    xs = []  # list to collect x sequences
    ys = []  # list to collect y sequences
    # iterate over data and create x, y sequences
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]  # get x sequence
        y = data[i + seq_length]  # get y sequence
        xs.append(x)
        ys.append(y)
    # return arrays of x sequences and y sequences
    return np.array(xs), np.array(ys)


# Setup Hyperparameters
input_dim = 1  # input dimension
hidden_dim = 48  # number of hidden layer dimensions
num_layers = 2  # number of hidden layers
output_dim = 1  # output dimension
seq_length = 15  # length of the sequence
num_epochs = 5000  # number of epochs for training

# Create sequences out of the data for time-series analysis
X, y = create_sequences(values_normalized, seq_length)

# Divide data into training and validation sets
train_size = int(len(X) * 0.9)
X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:]
y_val = y[train_size:]

# Convert data into PyTorch tensors for further processing
X_train = torch.FloatTensor(np.array(X_train)).view(-1, seq_length, input_dim).to(device)
y_train = torch.FloatTensor(np.array(y_train)).view(-1, output_dim).to(device)
X_val = torch.FloatTensor(np.array(X_val)).view(-1, seq_length, input_dim).to(device)
y_val = torch.FloatTensor(np.array(y_val)).view(-1, output_dim).to(device)

# Initialize the model, loss function, and optimizer
model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device)
criterion = torch.nn.MSELoss(reduction='mean')
# Use Mean Absolute Error (MAE) loss function for regression problem
mae_criterion = torch.nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training process
# Loop over each epoch for training
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Zero the gradients before running the backward pass

    # Forward pass: compute predicted y by passing x to the model
    y_train_pred = model(X_train)

    loss = criterion(y_train_pred, y_train)  # Compute and print loss
    loss.backward()  # Compute gradient of the loss with respect to model parameters
    optimizer.step()  # Calling step function on an Optimizer makes an update to its parameters

    # Periodically print loss
    if epoch % 10 == 0:
        print(f'Epoch {epoch} train loss: {loss.item()}')

# Model Validation
# Switch model to evaluation mode
model.eval()

# Predict the validation set results
y_val_pred = model(X_val)

# Inverse the normalization to get original values
y_val_pred_abs = scaler.inverse_transform(y_val_pred.detach().cpu().numpy())
y_val_true_abs = scaler.inverse_transform(y_val.detach().cpu().numpy())
y_train_pred_abs = scaler.inverse_transform(y_train_pred.detach().cpu().numpy())
y_train_true_abs = scaler.inverse_transform(y_train.detach().cpu().numpy())

# Compute Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Mean Relative Error (MRE)
# on the validation and train sets. These metrics provide us with measures of how well our model is performing.
val_rmse = root_mean_squared_error(y_val_true_abs, y_val_pred_abs)
val_mae = mean_absolute_error(y_val_true_abs, y_val_pred_abs)
val_relative = np.mean(np.abs(y_val_pred_abs / y_val_true_abs - 1))
train_rmse = root_mean_squared_error(y_train_true_abs, y_train_pred_abs)
train_mae = mean_absolute_error(y_train_true_abs, y_train_pred_abs)
train_relative = np.mean(np.abs(y_train_pred_abs / y_train_true_abs - 1))

# Print the RMSE, MAE, and MRE
print(f'Root Mean Squared Error (Validation): {val_rmse}')
print(f'Mean Absolute Error (Validation): {val_mae}')
print(f'Mean Relative Error (Validation): {val_relative}')
print(f'Root Mean Squared Error (Train): {train_rmse}')
print(f'Mean Absolute Error (Train): {train_mae}')
print(f'Mean Relative Error (Train): {train_relative}')

# Create a line plot of actual vs predicted values for our training set and validation set and enhances
# the plot with additional features like boundary lines and shaded regions.
# Setting the figure size
plt.figure(figsize=(10, 6))

# Creating date range for training and validation data
train_date_range = dates[seq_length: train_size + seq_length]
val_date_range = dates[train_size + seq_length:]

# Plotting the actual and predicted closing prices for train and validation set
plt.plot(train_date_range, y_train_true_abs, label='Actual Train', color='orange')
plt.plot(train_date_range, y_train_pred_abs, label='Predicted Train', color='red')
plt.plot(val_date_range, y_val_true_abs, label='Actual Validation', color='darkgreen')
plt.plot(val_date_range, y_val_pred_abs, label='Predicted Validation', color='blue')

# Define the boundary date as the last date in the training date range
boundary_date = train_date_range[-1]

# Draw a vertical line at the boundary date, making it evident where the training data ends and the validation data begins
plt.axvline(x=boundary_date, color='darkgreen', linestyle='--', lw=2, label='Train-Validation Boundary')

# Get the current x-axis limits for shading purposes
xlim_left, xlim_right = plt.xlim()

# Convert the boundary date from a Timestamp to a number recognized by matplotlib
boundary_date = mdates.date2num(boundary_date)

# Shade the area before the boundary in sky blue to represent the training data
plt.axvspan(xlim_left, boundary_date, color='skyblue', alpha=0.1)

# Shade the area after the boundary in light grey to represent the validation data
plt.axvspan(boundary_date, xlim_right, color='lightgrey', alpha=0.1)

# Give the plot a title and label the x and y axes
plt.title('Home Price Index Prediction using LSTM')
plt.xlabel('Date')
plt.ylabel('Home Price Index')
plt.legend()

# Add a legend to the plot for clarification on the color coding
plt.legend()

# Display the final plot
plt.show()
