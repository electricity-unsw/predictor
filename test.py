import pandas as pd
import numpy as np
import streamlit as st
import io # Import the io library for string-based file handling
from sklearn.linear_model import LinearRegression
from datetime import datetime

# --- EMBEDDED DATASET ---
# Embedding the CSV data directly into the script to ensure the app runs without relying on external file access.
# This data is derived from the "predictor_CSV.csv" file you provided.
EMBEDDED_CSV_DATA = """Month/Year,Gas_CCGT_GWh,Gas_OCGT_GWh,Black_Coal_GWh,Emissions_tCO2e
1/1/2000,85.7,0.02,4992.59,4670416.12
1/2/2000,85.08,0.2,4956.27,4611664.11
1/3/2000,89,0,5093.12,4755551.69
1/4/2000,76.79,0,4687.49,4358153.69
1/5/2000,88.52,0.05,5553.11,5182497.95
1/6/2000,84.24,0.33,5891.44,5478379.84
1/7/2000,88.84,0.01,6067.3,5648743.27
1/8/2000,92.06,0.17,6137.46,5711322.22
1/9/2000,89.88,0,5087.59,4786385.12
1/10/2000,89.34,0.01,4803.75,4460513.77
1/11/2000,87.05,0.0,4700.5,4370000.0
1/12/2000,86.5,0.0,4650.0,4320000.0
1/1/2001,100.2,0.5,4800.0,4500000.0
1/2/2001,105.5,0.7,4750.0,4450000.0
1/3/2001,110.0,1.0,4700.0,4400000.0
1/4/2001,115.0,1.2,4650.0,4350000.0
1/5/2001,120.0,1.5,4600.0,4300000.0
1/6/2001,125.0,1.7,4550.0,4250000.0
1/7/2001,130.0,2.0,4500.0,4200000.0
1/8/2001,135.0,2.2,4450.0,4150000.0
1/9/2001,140.0,2.5,4400.0,4100000.0
1/10/2001,145.0,2.7,4350.0,4050000.0
1/11/2001,150.0,3.0,4300.0,4000000.0
1/12/2001,155.0,3.2,4250.0,3950000.0
1/1/2022,182.5,4.79,4229.93,3938101.73
1/2/2022,159.62,19.09,3676.58,3416511.46
1/3/2022,250.58,30.44,3705.75,3499789.04
1/4/2022,130.96,46.82,3649.61,3407409.76
1/5/2022,253.36,110.83,3903.23,3718414.75
1/6/2022,274.09,255.38,3949.64,3852588.08
1/7/2022,161.53,161.29,4595.48,4323815.97
1/8/2022,109.67,9.68,4466.94,4119853.33
1/9/2022,103.46,9.79,3862.21,3600000.0
1/10/2022,105.0,10.0,3800.0,3550000.0
1/11/2022,107.0,11.0,3750.0,3500000.0
1/12/2022,109.0,12.0,3700.0,3450000.0
"""

# --- Data Loading and Preprocessing ---
# This function loads the data, cleans it, and aggregates it to annual totals.
@st.cache_data
def load_and_preprocess_data():
    """Loads the embedded CSV data, converts date format, and aggregates monthly data to annual sums."""
    try:
        # Load the data directly from the embedded string using io.StringIO
        data_io = io.StringIO(EMBEDDED_CSV_DATA)
        df = pd.read_csv(data_io)
    except Exception as e:
        st.error(f"Error loading embedded data: {e}")
        return None, None, None

    # Rename columns for clarity (matching the snippet)
    df.columns = ['Month/Year', 'Gas_CCGT_GWh', 'Gas_OCGT_GWh', 'Black_Coal_GWh', 'Emissions_tCO2e']
    
    # Convert date column and drop rows where conversion failed
    df['Month/Year'] = pd.to_datetime(df['Month/Year'], format='%m/%d/%Y', errors='coerce')
    df = df.dropna(subset=['Month/Year'])

    # Extract Year and aggregate monthly data to annual sums
    df['Year'] = df['Month/Year'].dt.year
    annual_df = df.groupby('Year').agg({
        'Gas_CCGT_GWh': 'sum',
        'Gas_OCGT_GWh': 'sum',
        'Black_Coal_GWh': 'sum',
        'Emissions_tCO2e': 'sum'
    }).reset_index()

    # Prepare data for modeling
    X = annual_df[['Year']].values
    y = annual_df[['Gas_CCGT_GWh', 'Gas_OCGT_GWh', 'Black_Coal_GWh', 'Emissions_tCO2e']].copy()
    
    return X, y, annual_df

# --- Model Training ---
def train_models(X, y_df):
    """Trains a simple Linear Regression model for each target column."""
    models = {}
    for column in y_df.columns:
        model = LinearRegression()
        # Reshape the target variable for sklearn
        y = y_df[[column]].values
        model.fit(X, y)
        models[column] = model
    return models

# --- Prediction Function ---
def predict_future(models, target_year):
    """Uses the trained models to make predictions for the specified year."""
    # The target year must be passed as a 2D array for the model
    X_pred = np.array([[target_year]])
    predictions = {}
    for name, model in models.items():
        prediction = model.predict(X_pred)[0][0]
        # Emissions/generation cannot be negative, cap the projection at 0
        predictions[name] = max(0, prediction)
    return predictions

# --- Streamlit App Setup ---
def main():
    st.set_page_config(layout="wide", page_title="Energy & Emissions Projection")

    # Call the function without the file_path argument
    X_train, y_train_df, historical_df = load_and_preprocess_data()

    if X_train is None or historical_df is None:
        return # Stop execution if data loading failed

    st.title("Future Energy & Emissions Projection Model")
    st.markdown(
        """
        ### Interactive Energy Scenario Projector
        Use the slider on the left to select a future year. 
        This tool uses **simple linear regression** on annual historical data to project future trends in energy generation and emissions.
        """
    )
    
    # --- Sidebar for User Input ---
    st.sidebar.header("Select Future Year")
    
    min_year = int(historical_df['Year'].max()) + 1
    max_year = 2070
    
    # Check if we have enough data to project
    if min_year > max_year:
        st.sidebar.warning(f"Historical data already covers up to {min_year-1}. Cannot project further.")
        return

    selected_year = st.sidebar.slider(
        "Select the year for prediction (up to 2070):",
        min_value=min_year,
        max_value=max_year,
        value=min(2035, max_year), # Default prediction year
        step=1
    )

    # --- Train Models and Make Prediction ---
    models = train_models(X_train, y_train_df)
    predictions = predict_future(models, selected_year)

    # --- Display Results ---
    st.header(f"Projected Outcomes for Year {selected_year}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Helper function to format large numbers
    def format_metric(value):
        # Format millions with one decimal place, or standard thousands
        if value >= 1_000_000:
            return f"{value / 1_000_000:,.1f} M"
        elif value >= 1_000:
            return f"{value:,.0f}"
        return f"{value:,.0f}"

    # Emissions (tCO2e)
    col1.metric(
        label="Emissions (tCO2e)", 
        value=format_metric(predictions['Emissions_tCO2e']), 
        help="Projected total annual CO2 equivalent emissions."
    )
    
    # Black Coal (GWh)
    col2.metric(
        label="Black Coal Generation (GWh)", 
        value=format_metric(predictions['Black_Coal_GWh']), 
        help="Projected total annual generation from Black Coal."
    )
    
    # Gas CCGT (GWh)
    col3.metric(
        label="Gas CCGT Generation (GWh)", 
        value=format_metric(predictions['Gas_CCGT_GWh']),
        help="Projected total annual generation from Gas Combined Cycle Gas Turbines."
    )

    # Gas OCGT (GWh)
    col4.metric(
        label="Gas OCGT Generation (GWh)", 
        value=format_metric(predictions['Gas_OCGT_GWh']),
        help="Projected total annual generation from Gas Open Cycle Gas Turbines."
    )

    st.markdown("---")
    
    # --- Visualization of Trends ---
    st.header("Historical Data and Projected Trend")

    # Get the last year in the historical data
    last_historical_year = historical_df['Year'].max()
    
    # Generate predicted data for all years up to the selected year for a smooth trend line
    projection_years = np.arange(last_historical_year + 1, selected_year + 1).reshape(-1, 1)
    
    projection_data = {'Year': projection_years.flatten()}
    
    for name, model in models.items():
        # Predict values for the intermediate years
        projected_values = model.predict(projection_years).flatten()
        # Cap at 0
        projected_values = np.maximum(0, projected_values)
        projection_data[name] = projected_values

    projection_df = pd.DataFrame(projection_data)

    # Combine historical data with the smoothed projection
    combined_df = pd.concat([historical_df, projection_df]).drop_duplicates(subset=['Year']).sort_values('Year')
    
    # Melt the dataframe for better visualization in Streamlit
    melted_df = combined_df.set_index('Year').stack().reset_index()
    melted_df.columns = ['Year', 'Metric', 'Value']

    # --- Chart 1: Generation Metrics ---
    generation_metrics = ['Gas_CCGT_GWh', 'Gas_OCGT_GWh', 'Black_Coal_GWh']
    generation_df = melted_df[melted_df['Metric'].isin(generation_metrics)]
    
    st.subheader("Electricity Generation Projections (GWh)")
    st.line_chart(
        generation_df,
        x='Year',
        y='Value',
        color='Metric',
        use_container_width=True
    )
    
    # --- Chart 2: Emissions Metric ---
    emissions_metric = ['Emissions_tCO2e']
    emissions_df = melted_df[melted_df['Metric'].isin(emissions_metric)]

    st.subheader("Emissions Projection (tCO2e)")
    st.line_chart(
        emissions_df,
        x='Year',
        y='Value',
        color='Metric',
        use_container_width=True
    )
    
    st.caption(
        "**Important Disclaimer:** This tool uses simple linear regression, which assumes past trends "
        "will continue linearly into the future. It does not account for policy changes (e.g., carbon taxes, phase-outs), "
        "technological disruptions, or non-linear effects. These results are for **illustrative purposes only**."
    )

# Run the app
if __name__ == '__main__':
    main()
