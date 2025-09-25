# COVID-19 Case Forecasting

This project documents a complete data analysis workflow, starting from raw data and culminating in a forecasting model. The goal is to process and understand COVID-19 case data to build a tool that can provide future projections. The journey begins with data ingestion and cleaning, moves through detailed exploration and visualization, and concludes with the application of a neural network to predict future trends.

## Tools & Libraries

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-2E8B57?style=for-the-badge&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-3B5526?style=for-the-badge&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/VS%20Code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white)

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

## Requirements & Usage

```bash
pip install pandas seaborn numpy matplotlib statsmodels scikit-learn tensorflow

# Usage
python covid_analysis.py

```
## File Structure

The project contains the following files:

- `README.md` — This file, containing project overview and instructions  
- `covid_analysis.py` — Main Python script for data analysis and forecasting  
- `json` — COVID_Dataset

## Contributing

This is a community project! You are welcome to:  

- Fork the repository  
- Add new features  
- Improve the existing code  

If you find any issues or have suggestions, please open an issue — I’d love to hear from you.
