# COVID-19 Case Forecasting

This project documents a complete data analysis workflow, starting from raw, unstructured data and culminating in a forecasting model. The goal is to process and understand COVID-19 case data to build a tool that can provide future projections. The journey begins with data ingestion and cleaning, moves through detailed exploration and visualization, and concludes with the application of a neural network to predict future trends.

## Data Source

The data used in this project is sourced from the COVID-19 Stream Data dataset, created by H. Gultekin and available on Kaggle.

- **Source:** [https://www.kaggle.com/datasets/hgultekin/covid19-stream-data/data](https://www.kaggle.com/datasets/hgultekin/covid19-stream-data/data)  
- **License:** Open Database License (ODbL)

## Features

- **Data Ingestion:** I start by reading and normalizing raw COVID-19 data from a JSON file, getting it ready for analysis.  

- **Exploratory Data Analysis (EDA):** This section examines the data's characteristics. This includes checking data types, handling missing values, and generating descriptive statistics to understand the dataset's contents.  

- **Data Visualization:** Charts are created to show the data. This includes:  
  - Bar charts showing top countries by cases and deaths.  
  - A pie chart to see the continental case distribution.  
  - Line charts to track monthly cases and a 7-day rolling average.  
  - Specialized plots like boxplots and QQ plots for outlier and normality detection.  

- **Time Series Analysis:** The data is prepared for a time series model by transforming it into a clean, daily series.  

- **Forecasting:** The project uses a Bidirectional LSTM neural network for forecasting. This model is applied for time series analysis to predict future cases.  

## Requirements

To run this script, you need to have the following Python libraries installed. You can easily get them all with one simple command:

```bash
pip install pandas numpy matplotlib statsmodels scikit-learn tensorflow


pip install pandas numpy matplotlib statsmodels scikit-learn tensorflow

Usage
Make sure your COVID-19 JSON data file is in the same directory as the script and is named json.

Open your terminal, navigate to the project folder, and run the script:

python covid_analysis.py

That's it! The script will automatically perform all the analysis steps and display the plots.

File Structure
README.md

covid_analysis.py

json (your data file)

Contributing
This is a community project! Feel free to fork the repository, add new features, or improve the existing code. If you find any issues or have suggestions, please open an issue—I'd love to hear from you.
