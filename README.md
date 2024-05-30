# Stock Market Analysis and Prediction Project

## Project Overview
This project aims to provide a comprehensive analysis of stock market data and develop predictive models to forecast stock prices and movements. It involves multiple steps including data preprocessing, exploratory data analysis, correlation analysis, sentiment analysis, volatility analysis, machine learning tasks, and reinforcement learning for portfolio optimization.

## Dataset
The dataset used in this project consists of historical stock market data for 10 companies over a significant period. The data includes daily stock prices and trading volumes, providing a rich source of information for analysis and modeling.

### Dataset Structure
The dataset is organized in a time series format with the following columns:

- **Company**: The name of the company (e.g., AAPL for Apple, NFLX for Netflix).
- **Date**: The date of the trading day in MM/DD/YYYY format.
- **Close/Last**: The closing price of the stock for that day.
- **Volume**: The number of shares traded during the day.
- **Open**: The opening price of the stock for that day.
- **High**: The highest price of the stock during the day.
- **Low**: The lowest price of the stock during the day.

### Example Data
Below is a small excerpt of the dataset to illustrate its structure:

| Company | Date       | Close/Last | Volume    | Open     | High     | Low      |
|---------|------------|------------|-----------|----------|----------|----------|
| AAPL    | 07/17/2023 | $193.99    | 50520160  | $191.90  | $194.32  | $191.81  |
| AAPL    | 07/14/2023 | $190.69    | 41616240  | $190.23  | $191.18  | $189.63  |
| AAPL    | 07/13/2023 | $190.54    | 41342340  | $190.50  | $191.19  | $189.78  |
| AAPL    | 07-12-2023 | $189.77    | 60750250  | $189.68  | $191.70  | $188.47  |
| AAPL    | 07-11-2023 | $188.08    | 46638120  | $189.16  | $189.30  | $186.60  |
| ...     | ...        | ...        | ...       | ...      | ...      | ...      |
| NFLX    | 07/24/2013 | $34.47     | 33395351  | $35.67   | $36.04   | $34.31   |
| NFLX    | 07/23/2013 | $35.75     | 76792963  | $35.91   | $37.46   | $35.17   |
| NFLX    | 07/22/2013 | $37.42     | 44791095  | $38.12   | $38.39   | $36.73   |
| NFLX    | 07/19/2013 | $37.80     | 18098750  | $38.20   | $38.28   | $37.60   |
| NFLX    | 07/18/2013 | $38.06     | 20418642  | $38.62   | $38.62   | $37.71   |

### Data Preprocessing Steps
1. **Loading the Data**: Import the dataset and inspect its structure.
2. **Date Conversion**: Convert the `Date` column to datetime format for consistency and ease of use.
3. **Removing Dollar Signs**: Remove dollar signs from numeric columns (e.g., `Close/Last`, `Open`, `High`, `Low`) and convert them to numeric types.
4. **Handling Missing Values**: Check for and appropriately handle any missing values.

## Usage
1. Clone the repository.
2. Open `Final_Project_Data_Analysis_AnomalyDetection_ML_DL.ipynb` script file and execute it using integrated development environment (IDE)
like Jupyter Notebook or Colaboratory. And upload Dataset into that platform so that it could initiate properly. !Notice
uploaded dataset `data_stocks_market.csv` into platform will be available only for 24 hours after that you will need to upload it again into that
platform.

## Steps

### Step 1: Loading the Data and Basic Preprocessing
- Load the stock market data.
- Perform basic preprocessing such as converting date formats and removing dollar signs from numeric columns.

### Step 2: Exploratory Data Analysis (EDA)
- **Line Plot for Stock Prices Over Time:**
  - Use Matplotlib to create line plots of stock prices (`Close/Last` column) over time for each company.
  - Iterate through each unique company in the dataset and plot its stock prices.
  - Add titles, labels, and legends for clarity.
- **Heatmap for Trading Volumes:**
  - Pivot the data to create a matrix where rows are dates, columns are companies, and values are trading volumes.
  - Use Seaborn to create a heatmap of trading volumes over time, highlighting periods of high and low activity.

### Step 3: Correlation Analysis
- **Pivot the Data for Correlation Analysis:**
  - Transform the data into a pivot table where rows represent dates, columns represent companies, and the values are the closing prices (`Close/Last`).
- **Calculate the Correlation Matrix:**
  - Compute the correlation matrix for the pivoted data, showing the pairwise correlation coefficients between the closing prices of different companies.
- **Plot the Correlation Matrix as a Heatmap:**
  - Use Seaborn to create a heatmap of the correlation matrix.
  - The heatmap visualizes the strength and direction of the relationships between the closing prices of various companies, with the color scale ranging from -1 (perfect negative correlation) to 1 (perfect positive correlation), and 0 indicating no correlation.

### Step 4: Top Performers Identification
- Identify the top-performing companies based on their stock price growth over the given time period.
- The data lists the percentage change in the stock prices of various companies, helping to identify which stocks performed best or worst over a specific period.

### Step 5: Market Sentiment Analysis
- **Data Preparation:**
  - Sample stock price data and news headlines were provided for illustration.
  - Convert the Date columns to datetime format for consistency.
- **Sentiment Calculation:**
  - Use the TextBlob library to calculate the sentiment polarity of news headlines.
  - Polarity scores range from -1 (negative sentiment) to 1 (positive sentiment).
- **Merge Data:**
  - Merge the sentiment data with stock data based on Company and Date.
  - Fill missing sentiment values with 0 (neutral sentiment).
- **Price Change Calculation:**
  - Calculate the daily percentage change in closing prices.
  - Drop rows with NaN values in the Price Change % column.
- **Plot the Impact of Sentiment on Stock Prices:**
  - Create a box plot to visualize the impact of sentiment on stock price changes.

### Step 6: Volatility Analysis
- Calculate the daily return (percentage change) of each stock.
- Use a 30-day rolling standard deviation to measure volatility.
- Remove any rows with missing volatility data.
- **Volatility Over Time:**
  - Plot the volatility of each company as a line over time, showing how the volatility of individual companies fluctuates.
- **Average Volatility:**
  - Plot the average volatility for each company as a bar chart, allowing easy comparison of the overall volatility of different companies.

### Step 7: Machine Learning Tasks
This step involves implementing various machine learning models to predict stock prices. Below are the models used:

1. **Model 1: LSTM with Hyperparameter Tuning using Keras Tuner**
   - This model utilizes Long Short-Term Memory (LSTM) networks with hyperparameter tuning performed using Keras Tuner's RandomSearch. The model is trained on the 'Close/Last' stock price data, which is normalized using MinMaxScaler. The sequences are created with a specified sequence length, and the model's performance is evaluated using metrics such as RMSE, MAE, and R-squared. The model is further optimized by finding the best hyperparameters and retraining it with these values.

2. **Model 2: LSTM with Hyperparameter Tuning using Keras Tuner (Shorter Tuning Duration)**
   - Similar to Model 1, this model also uses LSTM networks with Keras Tuner for hyperparameter optimization. However, the hyperparameter tuning is performed for a shorter duration, which may lead to quicker results but potentially less optimal hyperparameters. The training and evaluation process is similar, involving normalization, sequence creation, and performance metrics calculation.

3. **Model 3: LSTM with K-Fold Cross-Validation**
   - This model employs Long Short-Term Memory (LSTM) networks combined with K-Fold cross-validation to ensure robust evaluation. It includes the calculation of technical indicators such as Simple Moving Averages (SMA), Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD), along with time-based features and lag features. The data is normalized, and the model's performance is assessed using cross-validation, providing average metrics across multiple folds to ensure reliable evaluation. Metrics such as RMSE, MAE, and R-squared are used for performance evaluation.

### Step 8: Classification of Stock Movements
The code for this step:

1. **Data Loading and Preprocessing:**
   - Load historical stock data.
   - Preprocess the data, including cleaning and adding technical indicators.
   - Split the data into training and testing sets.

2. **Addressing Class Imbalance:**
   - Use SMOTE (Synthetic Minority Oversampling Technique) to handle class imbalance.

3. **Hyperparameter Tuning:**
   - Perform GridSearchCV to find the best hyperparameters for the logistic regression model.

4. **Model Evaluation:**
   - Evaluate the model's performance using various metrics such as accuracy, classification report, ROC curve, precision-recall curve, and feature importance.

### Step 9: Clustering Analysis
The code for this step involves:

1. **Data Preprocessing:**
   - Clean and convert data types.
   - Fill missing values.

2. **Feature Calculation:**
   - Calculate additional features like moving averages, Bollinger Bands, and Relative Strength Index (RSI) to capture trends.

3. **Standardization:**
   - Standardize features to ensure all features have equal weight in the clustering process.

4. **Optimal Number of Clusters:**
   - Use the Elbow Method to find the optimal number of clusters (k) based on inertia.

5. **K-Means Clustering:**
   - Perform k-means clustering with the chosen k to group data points into clusters.

6. **Cluster Analysis:**
   - Analyze cluster characteristics by calculating mean values for each feature within each cluster.

7. **Visualization:**
   - Use PCA to reduce dimensionality for a 2D scatter plot and visualize clusters.

### Step 10: Anomaly Detection
This step includes:

1. **Data Cleaning and Feature Engineering:**
   - Clean the data and engineer relevant features.

2. **Anomaly Detection:**
   - Implement anomaly detection techniques to identify unusual patterns in stock prices or trading volumes.

3. **Visualizations:**
   - Create visualizations to highlight detected anomalies.

### Step 11: Reinforcement Learning for Portfolio Optimization
The goal of this step is to optimize a stock trading strategy using reinforcement learning. Key objectives include:

1. **Policy Learning:**
   - The Reinforcement Learning (RL) model learns a policy that maps states (market conditions) to actions (trading decisions) to maximize expected return (cumulative reward).

2. **Portfolio Management Strategy:**
   - Develop an automated trading strategy that adapts to changing market conditions, making decisions on buying, holding, or selling assets.

3. **Performance Evaluation:**
   - Evaluate the model's performance by testing it on the same dataset and assessing the total reward accumulated during the testing phase.

Key steps in the code:

1. **Data Preprocessing:**
   - Load and prepare the financial dataset for training.

2. **Environment Creation:**
   - Set up a custom trading environment where the Reinforcement Learning (RL) agent can interact with the market data.

3. **Model Training:**
   - Train the Proximal Policy Optimization (PPO) model using the environment and specified hyperparameters.
   - Adjust `total_timesteps` to control the training duration and effectiveness.

4. **Model Evaluation:**
   - Test the trained model to evaluate its performance based on the total reward accumulated during the test phase.

Example training durations:
- `total_timesteps=100000` (1 hour 36 minutes): Extensive learning, likely high performance, long training time.
- `total_timesteps=10000` (20 minutes): Reasonable initial training, better performance, increased training time.
- `total_timesteps=1000` (12 minutes): Limited learning, possibly poor performance, short training time.

## Project Scope
This comprehensive set of steps covers the entire project scope from data analysis to machine learning tasks and deep learning. It is designed to provide insights into stock market behavior and build robust predictive models to assist in financial decision-making.
