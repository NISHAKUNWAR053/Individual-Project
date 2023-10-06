import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def timeseries_data(data, date_column):
    """
    Preprocesses time series data for analysis.
    
    Parameters:
        data (pd.DataFrame): The raw data.
        date_column (str): The column name containing the datetime values.
        
    Returns:
        data_clean (pd.DataFrame): The preprocessed data.
    """
    # Handling Missing Values in 'last_review'
    data_clean = data.dropna(subset=[date_column])
    
    # Convert 'last_review' to datetime data type
    data_clean[date_column] = pd.to_datetime(data_clean[date_column], format='%Y-%m-%d')
    
    # Extract year, month, and day for further analysis
    data_clean['year'] = data_clean[date_column].dt.year
    data_clean['month'] = data_clean[date_column].dt.month
    data_clean['day'] = data_clean[date_column].dt.day
    
    # Display the min and max date
    print(f"Date range: {data_clean[date_column].min()} to {data_clean[date_column].max()}")
    
    return data_clean



def plot_timeseries_data(data, date_column, value_column='number_of_reviews'):
    """
    Groups the data by date, plots the time series, and returns basic stats and a preview.
    
    Parameters:
        data (pd.DataFrame): The preprocessed data.
        date_column (str): The column name containing the datetime values.
        value_column (str): The column name where the count will be stored. Default is 'number_of_reviews'.
        
    Returns:
        time_series_data (pd.DataFrame): The time series data.
        (pd.DataFrame, pd.DataFrame): Basic stats and a preview of the time series data.
    """
    # Grouping by 'last_review' and counting the number of reviews for each date
    time_series_data = data.groupby(date_column).size().reset_index(name=value_column)

    # Plotting the time series data
    plt.figure(figsize=(14, 7))
    plt.plot(time_series_data[date_column], time_series_data[value_column], label='Number of Reviews')
    plt.title(f'Time Series of Airbnb Reviews in NYC ({time_series_data[date_column].min().year}-{time_series_data[date_column].max().year})')
    plt.xlabel('Date')
    plt.ylabel('Number of Reviews')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Returning basic stats and a preview of time_series_data for verification
    return time_series_data, (time_series_data.describe(), time_series_data.head())


import matplotlib.pyplot as plt
import statsmodels.api as sm

def decompose_and_plot_timeseries(data, date_column, value_column, seasonal_period=13):
    """
    Decomposes a time series into its components and plots them.
    
    Parameters:
        data (pd.DataFrame): The time series data.
        date_column (str): The column name containing the datetime values.
        value_column (str): The column name containing the values to decompose.
        seasonal_period (int): The seasonal period for the STL decomposition. Default is 13.
        
    Returns:
        decomposition (DecomposeResult): The decomposed time series components.
    """
    # Resampling the time series data to monthly frequency
    monthly_reviews = data.resample('M', on=date_column).sum()
    
    # Decomposing the time series into Trend, Seasonal, and Residual components using STL
    decomposition = sm.tsa.STL(monthly_reviews[value_column], seasonal=seasonal_period).fit()
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    # Plotting the original time series and its components
    plt.figure(figsize=(14, 10))
    plt.subplot(4, 1, 1)
    plt.plot(monthly_reviews[value_column], label='Original')
    plt.legend(loc='best')
    plt.title('Time Series Decomposition')
    
    plt.subplot(4, 1, 2)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.ylabel('Number of Reviews')
    
    plt.subplot(4, 1, 3)
    plt.plot(seasonal, label='Seasonal')
    plt.legend(loc='best')
    
    plt.subplot(4, 1, 4)
    plt.plot(residual, label='Residual')
    plt.legend(loc='best')
    plt.xlabel('Date')
    
    plt.tight_layout()
    plt.show()
    
    return decomposition


def stationarity_and_acf_pacf_plots(data, value_column):
    """
    Checks stationarity using Augmented Dickey-Fuller test and plots ACF and PACF.
    
    Parameters:
        data (pd.DataFrame): The time series data.
        value_column (str): The column name containing the values to analyze.
        
    Returns:
        adf_summary (pd.Series): The Augmented Dickey-Fuller test summary.
        is_stationary (bool): True if p-value <= 0.05, False otherwise.
    """
    # Stationarity Check using Augmented Dickey-Fuller test
    adf_result = adfuller(data[value_column])
    adf_summary = pd.Series(adf_result[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    
    # Plotting ACF and PACF
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    
    plot_acf(data[value_column], ax=ax[0])
    plot_pacf(data[value_column], ax=ax[1])
    
    plt.tight_layout()
    plt.show()
    
    # Returning ADF summary and stationarity check for verification
    return adf_summary, adf_result[1] <= 0.05  # True if p-value is less than 0.05


import matplotlib.pyplot as plt

def plot_annual_reviews_and_growth(data, date_column, value_column_name='number_of_reviews'):
    """
    Plots the annual number of reviews and growth rate.
    
    Parameters:
        data (pd.DataFrame): The time series data.
        date_column (str): The column name containing the datetime values.
        value_column_name (str): The name to be used for the aggregated value column. Default is 'number_of_reviews'.
        
    Returns:
        annual_reviews (pd.DataFrame): Data containing the annual number of reviews and growth rate.
    """
    # Extracting year from date_column
    data['year'] = data[date_column].dt.year
    
    # Grouping by year and summing the number of reviews
    annual_reviews = data.groupby('year').size().reset_index(name=value_column_name)
    
    # Calculating the annual growth rate
    annual_reviews['growth_rate'] = annual_reviews[value_column_name].pct_change() * 100
    
    # Plotting the annual reviews and growth rate
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plotting number of reviews
    ax1.set_title(f'Annual Number of Reviews and Growth Rate ({data[date_column].dt.year.min()}-{data[date_column].dt.year.max()})', fontsize=16)
    ax1.set_xlabel('Year', fontsize=14)
    ax1.set_ylabel('Number of Reviews', fontsize=14)
    ax1 = plt.plot(annual_reviews['year'], annual_reviews[value_column_name], marker='o', color='b')
    plt.xticks(annual_reviews['year'].unique())
    
    # Plotting growth rate
    ax2 = plt.gca().twinx()
    ax2.set_ylabel('Growth Rate (%)', fontsize=14)
    ax2 = plt.plot(annual_reviews['year'], annual_reviews['growth_rate'], marker='o', linestyle='dashed', color='r')
    
    # Adding legends
    plt.legend(['Growth Rate'], loc='upper left', bbox_to_anchor=(0.77, 0.97))
    plt.gca().add_artist(plt.legend(['Number of Reviews'], loc='upper left', bbox_to_anchor=(0.77, 1.0)))

    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Returning the annual_reviews data for verification
    return annual_reviews


import matplotlib.pyplot as plt

def plot_general_monthly_reviews(data, date_column, value_column_name='number_of_reviews'):
    """
    Plots the general monthly number of reviews (ignoring years).
    
    Parameters:
        data (pd.DataFrame): The time series data.
        date_column (str): The column name containing the datetime values.
        value_column_name (str): The name to be used for the aggregated value column. Default is 'number_of_reviews'.
        
    Returns:
        monthly_reviews_general (pd.DataFrame): Data containing the monthly number of reviews.
    """
    # Extracting month from date_column
    data['month'] = data[date_column].dt.month
    
    # Grouping by month (ignoring the year) to observe general monthly trends
    monthly_reviews_general = data.groupby('month').size().reset_index(name=value_column_name)
    
    # Plotting the monthly reviews
    plt.figure(figsize=(14, 7))
    plt.bar(monthly_reviews_general['month'], monthly_reviews_general[value_column_name], color='skyblue')
    plt.title('Monthly Number of Reviews (Aggregated Over Years)', fontsize=16)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Number of Reviews', fontsize=14)
    plt.xticks(range(1, 13), ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    
    # Returning the monthly_reviews_general data for verification
    return monthly_reviews_general


import matplotlib.pyplot as plt

def plot_weekday_reviews(data, date_column, value_column_name='number_of_reviews'):
    """
    Plots the number of reviews per weekday.
    
    Parameters:
        data (pd.DataFrame): The time series data.
        date_column (str): The column name containing the datetime values.
        value_column_name (str): The name to be used for the aggregated value column. Default is 'number_of_reviews'.
        
    Returns:
        weekday_reviews (pd.DataFrame): Data containing the number of reviews per weekday.
    """
    # Extracting the day of the week from 'last_review' and grouping by it
    data['weekday'] = data[date_column].dt.day_name()
    weekday_reviews = data.groupby('weekday').size().reset_index(name=value_column_name)
    
    # Ordering the weekdays properly
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_reviews = weekday_reviews.set_index('weekday').reindex(weekday_order).reset_index()
    
    # Plotting the reviews per weekday
    plt.figure(figsize=(14, 7))
    plt.bar(weekday_reviews['weekday'], weekday_reviews[value_column_name], color='salmon')
    plt.title('Number of Reviews per Weekday (Aggregated Over Years)', fontsize=16)
    plt.xlabel('Weekday', fontsize=14)
    plt.ylabel('Number of Reviews', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    
    # Returning the weekday_reviews data for verification
    return weekday_reviews






