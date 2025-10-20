import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 
import plotly.express as px 
from matplotlib.ticker import ScalarFormatter
import statsmodels.api as sm
import seaborn as sns 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, LSTM, Dropout, Bidirectional
from tensorflow.keras.regularizers import L1, L2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
"""
Hello! 

This project performs a Time Series Analysis of COVID-19 cases worldwide using a Deep Learning model (LSTM). 
The goal is to accurately predict the trend of COVID-19 cases, helping to understand and anticipate the evolution of the pandemic. 

The analysis includes:
- Exploratory data analysis (EDA)
- Data preprocessing and cleaning
- Feature engineering (lags, rolling statistics)
- Model training and evaluation using LSTM
- Visualization of predictions and trends

Project Steps: 

Exploratory Data Analysis

Checking the following: 
1- First and last 5 rows from the dataset
2- Column names 
3- Dataset Shape (number of rows and columns)
4- Columns Datatype
5- Total Missing Data per Column 
6- Descriptive statistics per Column (max, min, mean, std)
7- Value Counts per column (Countries, Cities, etc)

Applying the following: 
1- Grouping data based on specific column/s 
2- Visualizing Data in (Bar chart, Line chart, Pie chart)
3- Detecting Outliers using Boxplot 
4- Exploring Normality using QQ Plot 

Data Preprocessing 

Checking the following: 
1- Convert 'timestamp' column to proper datetime format
2- Set 'timestamp' as the DataFrame index 
3- Ensure the time frequency is consistent (hourly, daily, weekly, etc.)
4- Sort dataset chronologically
5- Check for duplicate timestamps
6- Detect missing timestamps in the sequence

Applying the following:
1- Resampling data to the required frequency 
2- Filling missing values 
3- Normalizing or standardizing numerical features 
4- Encoding categorical variables (if any)
5- Feature engineering 
6- Splitting dataset into training and testing sets while preserving time order

Model Preparation
1- Scale/normalize features 
2- Reshape data for LSTM input: [samples, timesteps, features]
3- Define LSTM architecture (layers, units, activation, dropout)
4- Compile the model (optimizer, loss, metrics)

Model Training
1- Train the model on the training set
2- Monitor performance on validation set & training set  
3- Use callbacks (EarlyStopping, ModelCheckpoint)

Model Evaluation
1- Evaluate model on test set (MAE, RMSE, MAPE)
2- Plot predictions vs actual values
3- Analyze residuals and errors

Model Deployment / Forecasting
1- Predict future values
2- Visualize forecasted trend
3- Save model for future use 

"""

#Import data from JSON file
data = pd.read_json("COVID_Dataset")

#Flatten the nested JSON data into a DataFrame
df = pd.json_normalize(data["records"])

#Perform Exploratory Data Analysis (EDA)
#Display the first 5 rows of the DataFrame
print(df.head(5))
#Display the last 5 rows of the DataFrame
print(df.tail(5))

#Print the column names
print(df.columns)

#Print the shape of the DataFrame (rows, columns)
print(df.shape)

#Print the data types of each column
print(df.dtypes)

#Count the number of missing values in each column
print(df.isnull().sum())

#Generate descriptive statistics for numerical columns
print(df[["cases", "deaths", "popData2019", "Cumulative_number_for_14_days_of_COVID-19_cases_per_100000" ]].describe())

#Generate descriptive statistics for object (categorical) columns
print(df[["countriesAndTerritories", "continentExp"]].describe(include="object"))

#Count the occurrences of each unique value in the 'continentExp' column
print(df["continentExp"].value_counts())

#Count the occurrences of each unique value in the 'countriesAndTerritories' column
print(df["countriesAndTerritories"].value_counts())

#Group data by continent, country, and population to sum cases and deaths, then sort
df_grouped_by_continent = df.groupby(["continentExp", "countriesAndTerritories", "popData2019"])[["cases", "deaths"]].sum().sort_values("cases", ascending=False).reset_index()

#Display the top 5 rows of the grouped data
print(df_grouped_by_continent.head(5))

#Select the top 15 countries for plotting
df_grouped_by_continent_top_15 = df_grouped_by_continent.head(15)

#Create a bar chart for the top 15 countries' total COVID cases
plt.bar(df_grouped_by_continent_top_15['countriesAndTerritories'],df_grouped_by_continent_top_15['cases']/1000000, label='Cases')

#Add labels and a title to the plot
plt.ylabel('Total')
plt.title('Top 15 Countries in total COVID Cases - Million')
plt.xticks(rotation=90)
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.ticklabel_format(axis='y', style='plain')
plt.legend()
plt.show()

#Create a bar chart for the top 15 countries' total COVID deaths
plt.bar(df_grouped_by_continent_top_15['countriesAndTerritories'], df_grouped_by_continent_top_15['deaths']/1000, label='deaths')

#Add labels and a title to the plot
plt.ylabel('Total')
plt.title('Top 15 Countries in total COVID Deaths - Thousands')
plt.xticks(rotation=90)
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.ticklabel_format(axis='y', style='plain')
plt.legend()
plt.show()

#Create a pie chart for the top 5 countries' cases
df_grouped_by_continent_top_5 = df_grouped_by_continent.head(5)
plt.pie(df_grouped_by_continent_top_5["cases"].astype(int), labels=df_grouped_by_continent_top_5['countriesAndTerritories'], autopct="%1.1f%%")
plt.show()

#Convert the 'dateRep' column to datetime objects
df["dateRep"] = pd.to_datetime(df["dateRep"], format="%d/%m/%Y")

#Set 'dateRep' as the index
df = df.set_index("dateRep")

#Sort the DataFrame by index in ascending order
df = df.sort_index(ascending=True)

#Create a line chart showing total cases by month
df_total_by_month = df[["cases", "deaths"]].resample("M").sum() 
plt.plot(df_total_by_month.index, df_total_by_month["cases"])
plt.legend()
plt.xlabel("Date")
plt.ylabel("Total Cases By Month")
plt.title("Total New Cases Per Month")
plt.show()

#Detect outliers in the 'cases' column using a boxplot
plt.boxplot(df["cases"])
plt.legend()
plt.show()

#Detect normality of the 'cases' column using a QQ Plot
fig = sm.qqplot(df["cases"], line='45')
plt.show()

#Explore time series trend
#Calculate a 7-day rolling average for cases
df['cases_rolling'] = df['cases'].rolling(window=7).mean()

#Plot the daily cases and the 7-day rolling average
plt.plot(df.index, df['cases'], alpha=0.4, label='Daily cases')
plt.plot(df.index, df['cases_rolling'], color='red', label='7-day rolling avg')
plt.legend()
plt.show()

#Group data by continent, country, and population to sum cases and deaths, then sort (duplicate)
df_grouped_by_continent = df.groupby(["continentExp", "countriesAndTerritories", "popData2019"])[["cases", "deaths"]].sum().sort_values("cases", ascending=False).reset_index()

#Create a copy of the DataFrame
df1 = df.copy() 

#Create a unique timestep by grouping and summing cases and deaths per day
df_daily = df.groupby('dateRep').agg({'cases':'sum','deaths':'sum'})

#Fill in missing dates with zero values
full_range = pd.date_range(df_daily.index.min(), df_daily.index.max(), freq='D')
df_daily = df_daily.reindex(full_range).fillna(0).rename_axis('dateRep')

#Calculate the difference between consecutive dates
df_daily['date_diff'] = df_daily.index.diff()

#Print the value counts of 'date_diff'
print(df_daily['date_diff'].value_counts())

#Drop the 'date_diff' column
df_daily.drop(columns="date_diff", inplace=True)

#Print the first 3 rows of the daily DataFrame
print(df_daily.head(3))

#Print the column names of the daily DataFrame
print(df_daily.columns)

#Print the shape of the daily DataFrame
print(df_daily.shape)

#Splitting the dataset into training and testing sets
#Drop the 'deaths' column
df_daily_cases = df_daily.drop(columns="deaths")

#Calculate the training and testing set sizes
train_size = int(len(df_daily_cases)* 0.80)
test_size = int(len(df_daily_cases) - train_size)

#Slice the data into training and testing sets
train_cases = df_daily_cases.iloc[0:train_size]
test_cases = df_daily_cases.iloc[train_size:len(df_daily_cases)]

#Define batch size, length, and features
batch_size = 10
length = 12
features = 2

#Initialize the MinMaxScaler
Scaler = MinMaxScaler()

#Fit and transform the training data, then transform the testing data
train_sc = pd.DataFrame(Scaler.fit_transform(train_cases), columns=train_cases.columns)
test_sc = pd.DataFrame(Scaler.transform(test_cases), columns=test_cases.columns) 

#Define a function to create a dataset for time series forecasting
def create_dataset(X, y, time_steps=1):
  Xs, ys = [] , []

  for i in range(len(X) - time_steps):
    v = X.iloc[i:(i + time_steps)].values
    Xs.append(v)
    ys.append(y.iloc[i + time_steps])
  return np.array(Xs), np.array(ys)

#Define time steps
time_steps=12

#Create the training and testing datasets using the function
X_train, y_train = create_dataset(train_sc, train_sc, time_steps)

X_test, y_test = create_dataset(test_sc, test_sc, time_steps)

#Build the deep learning model
#Initialize a Sequential model
Model = Sequential()

#Define an EarlyStopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

#Add a Bidirectional LSTM layer
Model.add(Bidirectional(LSTM(60, input_shape=(X_train.shape[1], X_train.shape[2])))) 
#Add a Dense output layer
Model.add(Dense(1))

#Compile the model with Adam optimizer and mean squared error loss
Model.compile(optimizer="adam", loss="mse")

#Train the model
his = Model.fit(X_train, y_train, epochs=30, shuffle=False, validation_split=0.2, batch_size=10, callbacks=[early_stop], verbose=1)

#Plot the training and validation loss
plt.plot(his.history["loss"], color="r", label="loss")
plt.plot(his.history["val_loss"], color="b", label="val loss")
plt.legend()
plt.show()

#Make predictions on the testing data
y_pred = Model.predict(X_test)

#Inverse transform the scaled data to get the original values
y_test_inv = Scaler.inverse_transform(y_test.reshape(1,-1))

y_train_inv = Scaler.inverse_transform(y_train.reshape(1,-1))

y_pred_inv = Scaler.inverse_transform(y_pred)

#Plot the true values vs. the predicted values
plt.figure(figsize=(12,6))
plt.plot(y_test_inv.flatten(), color='r', label='True')
plt.plot(y_pred_inv, color='b', label='Predicted')
plt.title("Predicted vs True Values")
plt.xlabel("Time")
plt.ylabel("Cases")
plt.legend()
plt.show()
