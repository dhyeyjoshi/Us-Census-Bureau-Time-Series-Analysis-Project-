import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


# Ensure TensorFlow uses only CPU
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load and preprocess the data
@st.cache_data
def load_data(filepath):
    data = pd.read_csv(filepath, index_col='year', parse_dates=True)
    data.index = pd.to_datetime(data.index, format='%Y')
    data = data.asfreq('YS-JAN')
    return data

@st.cache_data
def preprocess_data(filepath):
    sal_data = pd.read_csv(filepath, index_col='year', parse_dates=True)
    sal_data_log = np.log(sal_data['total_earners_salary'])
    sal_data_log.index = pd.to_datetime(sal_data_log.index, format='%Y')
    sal_data_log = sal_data_log.asfreq('YS-JAN')
    sal_data_log.dropna(inplace=True)
    return sal_data_log

data_path = './data/salary_clean.csv'
sal_data_log = preprocess_data(data_path)
original_data = load_data(data_path)

# Assuming 'year' is in the index, we create a 'year' column
original_data['year'] = original_data.index.year

# Sidebar with settings
st.sidebar.title("Settings")

st.sidebar.header("Analyze Salaries By Education Level")
st.sidebar.markdown("Select Education Level to visualize the salaries over the years:")

# Sidebar - Salaries by Education Level
education_levels = {
    'no_high_school_salary': 'With No High School Degree',
    'high_school_salary': 'With High School Degree',
    'some_college_salary': 'With Some College Degree',
    'bachelors_salary': 'With Bachelors Degree',
    'adv_salary': 'With Advanced Degree'
}
selected_salaries = []
for key, value in education_levels.items():
    if st.sidebar.checkbox(value, key=key):
        selected_salaries.append(key)

# Sidebar - Year picker for the pie chart
st.sidebar.header("Analyze Educational Segmentation")

year = st.sidebar.selectbox('Select Year to Analyze Educational Segmentation:', list(range(1975, 2022)), index=len(range(1975, 2022))-1)

# Function to create pie chart
def create_pie_chart(data, year):
    # Filter the data for the selected year
    year_data = data.loc[data.index.year == year, education_levels.keys()].sum()
    fig = px.pie(
        names=education_levels.values(),
        values=year_data,
        title=f'Proportion of Total Earners by Education Level in {year}'
    )
    return fig

# Sidebar - 3D Plot Settings
st.sidebar.header("Analyzing Trends Over Time")
st.sidebar.markdown("Select columns for X and Y axes to visualize the dynamic relationship between different metrics across years:")
# Available columns for X and Y axis
available_columns = original_data.select_dtypes(include=[np.number]).columns.tolist()
x_axis = st.sidebar.selectbox('Choose X axis', available_columns, index=0)
y_axis = st.sidebar.selectbox('Choose Y axis', available_columns, index=1)

# Make sure 'year' is a column in the dataframe
if 'year' not in original_data.columns:
    original_data['year'] = original_data.index.year

# Function to create 3D scatter plot with year on Z axis
def create_3d_scatter_plot(data, x, y):
    fig = go.Figure(data=[go.Scatter3d(
        x=data[x],
        y=data[y],
        z=data['year'],
        mode='markers',
        marker=dict(
            size=5,
            color=data['year'],  # Color points by year for a gradient effect
            colorscale='Viridis',  # Choose a colorscale
            opacity=0.8,
            colorbar_title='Year'
        )
    )])
    
    # Adjust layout to make sure title and other plot elements are visible
    fig.update_layout(
        title={
            'text': f'Interactive 3D Visualization of Yearly Trends in {x} vs. {y}',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        scene=dict(
            xaxis_title=x,
            yaxis_title=y,
            zaxis_title='Year',
            # Adjusting the camera view angle if necessary
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.5)
            )
        ),
        margin=dict(l=10, r=10, b=10, t=50)  # Adjusting margins
    )
    return fig

# Sidebar - Select original or log-transformed data for decomposition
st.sidebar.subheader("Decomposition Selection")
use_log_data = st.sidebar.checkbox('Use Log Transformed Data for Decomposition')
use_original_data_for_decomp = st.sidebar.checkbox('Use Original Data for Decomposition')

# Ensure that only one checkbox can be checked
if use_log_data and use_original_data_for_decomp:
    st.sidebar.error("Please select only one option for decomposition.")
elif not use_log_data and not use_original_data_for_decomp:
    st.sidebar.error("Please select an option for decomposition.")

# Function to decompose the time series and plot the components
def plot_decomposition(data, log_transform):
    # Perform log transformation if selected
    if log_transform:
        data_to_decompose = np.log(data.replace(0, np.nan)).dropna()
    else:
        data_to_decompose = data.replace(0, np.nan).dropna()
    
    # Decompose
    decomposition = seasonal_decompose(data_to_decompose, model='additive', period=12)
    
    # Plot
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(decomposition.observed, label='Original')
    axs[0].set_title('Observed')
    axs[0].legend()
    
    axs[1].plot(decomposition.trend, label='Trend')
    axs[1].set_title('Trend')
    axs[1].legend()
    
    axs[2].plot(decomposition.seasonal, label='Seasonal')
    axs[2].set_title('Seasonality')
    axs[2].legend()
    
    axs[3].plot(decomposition.resid, label='Residual')
    axs[3].set_title('Residuals')
    axs[3].legend()
    
    plt.tight_layout()
    fig.suptitle('Decomposing Time Series into its components: Trend, Seasonality and Residual', weight='bold', y=1.05, fontsize='large')
    return fig

# Load and preprocess the data
@st.cache_data
def load_data(filepath):
    data = pd.read_csv(filepath, index_col='year', parse_dates=True)
    data.index = pd.to_datetime(data.index, format='%Y')
    data = data.asfreq('YS-JAN')
    data['year'] = data.index.year 
    return data

@st.cache_data
def preprocess_data(filepath):
    sal_data = pd.read_csv(filepath, index_col='year', parse_dates=True)
    sal_data_log = np.log(sal_data['total_earners_salary'])
    sal_data_log.index = pd.to_datetime(sal_data_log.index, format='%Y')
    sal_data_log = sal_data_log.asfreq('YS-JAN')
    sal_data_log.dropna(inplace=True)
    return sal_data_log

data_path = './data/salary_clean.csv'
sal_data_log = preprocess_data(data_path)
original_data = load_data(data_path)

# Fit the selected model
def fit_model(data, model_type):
    if model_type == 'ARIMA':
        model = ARIMA(data, order=(2, 1, 2))
    elif model_type == 'SARIMAX':
        model = SARIMAX(data, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12))
    elif model_type == 'AR':
        model = ARIMA(data, order=(2, 0, 0))
    elif model_type == 'MA':
        model = ARIMA(data, order=(0, 0, 2))
    elif model_type == 'Naive':
        return None  # No model fitting needed for Naive approach
    else:
        return None
    return model.fit()

# Streamlit UI
st.title("US Census Bureau: Time Series Analysis On Average Salary Growth Every Year")

# Display the first few rows of the dataframe
st.write("Here's a glimpse of the dataset:")
st.dataframe(original_data.head())

# Model selection
selected_models = st.multiselect(
    "Select Model(s)",
    ['ARIMA', 'SARIMAX', 'AR', 'MA', 'Naive'],
    ['ARIMA']
)

# Forecast slider
forecast_years = st.slider('Select forecast years', 1, 20, 5)

# Update graph and metrics
def update_graph_and_metrics(data, selected_models, forecast_years):
    fig = go.Figure()
    error_metrics = []
    forecast_data = pd.DataFrame()
    for model_type in selected_models:
        if model_type == 'Naive':
            last_value = data.iloc[-1]
            forecast_values = np.array([last_value] * forecast_years)
            forecast_index = pd.date_range(data.index[-1], periods=forecast_years + 1, freq='YS-JAN')[1:]
            forecast_df = pd.DataFrame(np.exp(forecast_values), index=forecast_index, columns=['Forecast'])
            mse = mean_squared_error([np.exp(last_value)] * forecast_years, forecast_values)
            mae = mean_absolute_error([np.exp(last_value)] * forecast_years, forecast_values)
        else:
            fitted_model = fit_model(data, model_type)
            forecast = fitted_model.get_forecast(steps=forecast_years)
            forecast_values = forecast.predicted_mean.values
            forecast_index = pd.date_range(data.index[-1], periods=forecast_years + 1, freq='YS-JAN')[1:]
            forecast_df = pd.DataFrame(np.exp(forecast_values), index=forecast_index, columns=['Forecast'])
            mse = mean_squared_error(np.exp(data[-forecast_years:]), forecast_values)
            mae = mean_absolute_error(np.exp(data[-forecast_years:]), forecast_values)
        
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name=f'{model_type} Forecast'))
        error_metrics.append(f'{model_type} - MSE: {mse:.2f}, MAE: {mae:.2f}')
        forecast_data = pd.concat([forecast_data, forecast_df.assign(Model=model_type)], axis=0)
    
    fig.add_trace(go.Scatter(x=data.index, y=np.exp(data), mode='lines', name='Original Salary'))
    fig.update_layout(title='Time Series Forecast of Total Earners Salary', xaxis_title='Year', yaxis_title='Salary')
    
    return fig, error_metrics, forecast_data

if selected_models:
    fig, error_metrics, forecast_data = update_graph_and_metrics(sal_data_log, selected_models, forecast_years)
    st.plotly_chart(fig)
    for metric in error_metrics:
        st.write(metric)
    
    # Display the forecasted table
    st.write("Forecasted Data:")
    st.dataframe(forecast_data.reset_index().rename(columns={'index': 'Date'}))  # Reset index to display Date as a column

    # Download button
    st.download_button("Download forecast data", forecast_data.to_csv().encode('utf-8'), "forecast_data.csv")
    
# Dynamic plot based on selected salaries
def plot_selected_salaries(data, selected_salaries):
    fig = go.Figure()
    for salary in selected_salaries:
        if salary in data:
            fig.add_trace(go.Scatter(x=data.index, y=data[salary], mode='lines', name=education_levels[salary]))
    fig.update_layout(title='Salaries by Education Level Over the Years', xaxis_title='Year', yaxis_title='Salary')
    return fig

# Display dynamic plot
if selected_salaries:
    salary_fig = plot_selected_salaries(original_data, selected_salaries)
    st.plotly_chart(salary_fig)
    

if 'year' in locals():
    pie_fig = create_pie_chart(original_data, year)
    st.plotly_chart(pie_fig)
    
# Display the 3D scatter plot in the main page
if x_axis and y_axis:
    scatter_3d_fig = create_3d_scatter_plot(original_data, x_axis, y_axis)
    st.plotly_chart(scatter_3d_fig)
    

# Main - Perform decomposition based on the sidebar selection
if use_log_data or use_original_data_for_decomp:
    # Assuming we are decomposing the 'total_earners_salary' column
    # Replace with the appropriate column name if needed
    decomposed_fig = plot_decomposition(original_data['total_earners_salary'], use_log_data)
    st.pyplot(decomposed_fig)
    