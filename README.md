# Boston Housing Price Prediction using RNN, LSTM, and GRU

This project implements and compares three deep learning recurrent neural network models for predicting house prices using the Boston Housing dataset. The notebook prepares the dataset as sequential input data and trains **Simple RNN**, **LSTM**, and **GRU** models for regression.

## Project Overview

The main objective of this project is to evaluate how different recurrent neural network architectures perform on a regression problem. The Boston Housing dataset contains housing-related numerical features, and the target variable represents house price values.

Although RNN-based models are more commonly used for sequential/time-series data, this project converts the dataset into sequences to compare the behavior of RNN, LSTM, and GRU models.

## Dataset

The project uses the **Boston Housing dataset** available through `tf.keras.datasets`.

Dataset details:

- Total samples: **506**
- Number of input features: **13**
- Target variable: Median house value
- Task type: Regression

Initial dataset shapes:

| Data | Shape |
|---|---|
| Raw training features | `(404, 13)` |
| Raw training target | `(404,)` |
| Raw test features | `(102, 13)` |
| Raw test target | `(102,)` |
| Complete dataset | `(506, 13)` |

## Data Preprocessing

The following preprocessing steps are performed:

1. Load the Boston Housing dataset.
2. Combine training and testing data.
3. Normalize feature values using `StandardScaler`.
4. Normalize target values using `StandardScaler`.
5. Convert the dataset into sequential format.
6. Split the sequence data into training, validation, and testing sets.

Sequence configuration:

- Sequence length: **10**
- Input sequence shape: **(496, 10, 13)**
- Output target shape: **(496,)**

Final data split:

| Split | Shape |
|---|---|
| Training data | `(347, 10, 13)` |
| Validation data | `(74, 10, 13)` |
| Testing data | `(75, 10, 13)` |

## Models Implemented

Three recurrent neural network models are trained and compared.

### 1. Simple RNN

The Simple RNN model uses a basic recurrent layer followed by dense layers.

Architecture:

- SimpleRNN layer with 64 units
- Dense layer with 32 units
- Output Dense layer with 1 unit

### 2. LSTM

The LSTM model is used to capture longer-term dependencies in sequential data.

Architecture:

- LSTM layer with 64 units
- Dense layer with 32 units
- Output Dense layer with 1 unit

### 3. GRU

The GRU model is a lighter recurrent architecture compared to LSTM and is used for sequence learning.

Architecture:

- GRU layer with 64 units
- Dense layer with 32 units
- Output Dense layer with 1 unit

## Training Configuration

All models are trained using the same configuration:

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Mean Squared Error |
| Metric | Mean Absolute Error |
| Epochs | 30 |
| Batch Size | 16 |

## Results

The models were evaluated on the test dataset using MSE, MAE, RMSE, R² score, and training time.

| Model | Test Loss MSE | MAE Original Scale | RMSE Original Scale | R² Score | Training Time |
|---|---:|---:|---:|---:|---:|
| Simple RNN | 1.7832 | 9.3765 | 12.2695 | -0.6448 | 6.65 sec |
| LSTM | 2.1647 | 10.8220 | 13.5183 | -0.9966 | 8.56 sec |
| GRU | 1.6050 | 8.7788 | 11.6402 | -0.4804 | 8.97 sec |

## Best Model

Based on the results:

- **Best model based on Test Loss:** GRU
- **Best model based on RMSE:** GRU
- **Best model based on R² Score:** GRU
- **Fastest model:** Simple RNN

The **GRU model** performed best overall because it achieved the lowest test loss, lowest RMSE, and highest R² score among the three models.

## Visualizations

The notebook includes the following visualizations:

- Sample sequential target values
- RNN training and validation loss curve
- LSTM training and validation loss curve
- GRU training and validation loss curve
- Model comparison based on Test MAE
- Actual vs predicted values for RNN
- Actual vs predicted values for LSTM
- Actual vs predicted values for GRU
- Validation loss comparison of all models

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- TensorFlow
- Keras
- Scikit-learn

## How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repository-name.git
```

2. Open the notebook:

```bash
jupyter notebook
```

3. Install the required libraries if needed:

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn
```

4. Run all cells in the notebook.

## Project Structure

```text
.
├── dl-assignment-2.ipynb
└── README.md
```

## Key Learning Outcomes

This project demonstrates:

- Loading a built-in regression dataset from TensorFlow/Keras
- Normalizing features and target values
- Creating sequential input data
- Building RNN, LSTM, and GRU models
- Training recurrent neural networks for regression
- Comparing models using MSE, MAE, RMSE, R² score, and training time
- Visualizing training performance and prediction results

## Conclusion

This project compares Simple RNN, LSTM, and GRU models for Boston Housing price prediction. Among the three models, **GRU achieved the best overall performance** on the test set. However, the negative R² scores show that the models did not generalize strongly, which means further improvement is possible through better feature engineering, hyperparameter tuning, and using model architectures more suitable for tabular regression data.
