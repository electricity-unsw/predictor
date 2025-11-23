import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from datetime import datetime

# --- Data Loading and Preprocessing ---
# This function loads the data, cleans it, and aggregates it to annual totals.
@st.cache_data
def load_and_preprocess_data(file_path):
    """Loads the CSV, converts date format, and aggregates monthly data to annual sums."""
    try:
        # Load the monthly data using the provided file path
        # In a deployed environment, Streamlit handles file paths based on uploads.
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        # Fallback for environments where the file might not be accessible directly
        st.error("Dataset not found. Please ensure 'predictor_CSV.csv' is in the correct location or uploaded.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
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

    # The file path provided by the Canvas environment for the uploaded file
    file_path = "predictor_CSV.csv" 
    
    X_train, y_train_df, historical_df = load_and_preprocess_data(file_path)

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
