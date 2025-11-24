import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px
import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Energy & Emissions Predictor",
    page_icon="⚡",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #262730;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #41424b;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #4CAF50;
    }
    .metric-label {
        font-size: 14px;
        color: #FAFAFA;
    }
    h1, h2, h3 {
        color: #FAFAFA;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    # Attempt to load the file - checks for standard name or specific upload name
    try:
        # Priority: Look for the clean filename
        df = pd.read_csv("predictor_csv.csv")
    except FileNotFoundError:
        try:
            # Fallback: Look for the original upload name
            df = pd.read_csv("predictor_csv.xlsx - Sheet1.csv")
        except:
            st.error("Data file not found. Please ensure 'predictor_csv.csv' is in the directory.")
            return None

    # Clean and Parse Dates
    df['date'] = pd.to_datetime(df['date'])
    
    # Create a numeric representation of date for Regression (Ordinal)
    df['date_ordinal'] = df['date'].apply(lambda x: x.toordinal())
    
    return df

df = load_data()

if df is not None:
    # --- Sidebar Controls ---
    st.sidebar.header("Prediction Controls")
    
    # Slider for Year Selection (2025 - 2070)
    target_year = st.sidebar.slider("Select Target Year", 2025, 2070, 2030)
    
    # Option to toggle between User Equations and Dynamic fit for Generation
    # We use Dynamic for Dates by default because Excel date serials differ from Python ordinals
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Note on Logic:**\n\n"
        "1. **Date vs Gen:** Uses Python Linear Regression on historical data (to ensure correct time-axis alignment).\n"
        "2. **Gen vs Emission:** Uses the specific Excel formulas provided in your prompt."
    )

    # --- Regression Logic ---

    # 1. Train Models: Date (Independent) -> Generation (Dependent)
    # We need to train these to get the slope/intercept for the Time axis
    
    models_gen = {}
    
    # Define the mapping of column names
    targets = {
        'Coal': {
            'gen_col': 'Coal (Black) -  GWh',
            'em_col': 'Coal (Black) Emissions Vol - tCO₂e',
            'color': '#ff4b4b' # Red
        },
        'Gas CCGT': {
            'gen_col': 'Gas (CCGT) -  GWh',
            'em_col': 'Gas (CCGT) Emissions Vol - tCO₂e',
            'color': '#ffa500' # Orange
        },
        'Gas OCGT': {
            'gen_col': 'Gas (OCGT) -  GWh',
            'em_col': 'Gas (OCGT) Emissions Vol - tCO₂e',
            'color': '#00c0f2' # Blue
        }
    }

    # Train Generation Models
    X = df[['date_ordinal']]
    
    for key, val in targets.items():
        y = df[val['gen_col']]
        model = LinearRegression()
        model.fit(X, y)
        models_gen[key] = model

    # --- Prediction Function ---
    def predict_values(year):
        # Create a date object for the middle of the requested year
        target_date = datetime.date(year, 6, 1) 
        target_ordinal = target_date.toordinal()
        
        predictions = {}
        total_emissions = 0
        total_gen = 0
        
        for key, val in targets.items():
            # 1. Predict Generation based on Year (Using Python Model)
            gen_pred = models_gen[key].predict([[target_ordinal]])[0]
            
            # 2. Predict Emissions based on Generation (Using USER PROVIDED EQUATIONS)
            # Coal: y = 958.45x - 185539
            # CCGT: y = 413.21x + 14159
            # OCGT: y = 582.73x - 114.71
            
            if key == 'Coal':
                em_pred = (958.45 * gen_pred) - 185539
            elif key == 'Gas CCGT':
                em_pred = (413.21 * gen_pred) + 14159
            elif key == 'Gas OCGT':
                em_pred = (582.73 * gen_pred) - 114.71
            
            # Handle negative predictions (physically impossible) by clamping to 0
            if gen_pred < 0: gen_pred = 0
            if em_pred < 0: em_pred = 0
            
            predictions[key] = {
                'gen': gen_pred,
                'em': em_pred
            }
            
            total_gen += gen_pred
            total_emissions += em_pred
            
        return predictions, total_gen, total_emissions

    # Calculate for current selection
    preds, tot_gen, tot_em = predict_values(target_year)

    # --- Dashboard Layout ---
    
    st.title("Energy Forecast Dashboard")
    st.markdown(f"### Projections for Year: **{target_year}**")

    # Top Level Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Predicted Generation</div>
            <div class="metric-value">{tot_gen:,.2f} GWh</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Predicted Emissions</div>
            <div class="metric-value">{tot_em:,.2f} tCO₂e</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Breakdown by Source")
    
    # Detailed Columns
    c1, c2, c3 = st.columns(3)
    
    sources = ['Coal', 'Gas CCGT', 'Gas OCGT']
    cols = [c1, c2, c3]
    
    for source, col in zip(sources, cols):
        with col:
            st.markdown(f"#### {source}")
            st.metric("Generation (GWh)", f"{preds[source]['gen']:,.2f}")
            st.metric("Emissions (tCO₂e)", f"{preds[source]['em']:,.2f}")

    st.divider()

    # --- Tabs for Analysis ---
    tab1, tab2, tab3 = st.tabs(["📉 Generation Trends", "☁️ Emission Trends", "📊 Linear Regression Validation"])

    # Prepare Future Data for plotting (Line from last data point to 2070)
    last_date = df['date'].max()
    future_years = list(range(last_date.year, 2071))
    future_dates = [datetime.date(y, 1, 1) for y in future_years]
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)

    with tab1:
        st.subheader("Generation Forecast (GWh)")
        fig_gen = go.Figure()

        for key, val in targets.items():
            # Historical Dots
            fig_gen.add_trace(go.Scatter(
                x=df['date'], 
                y=df[val['gen_col']], 
                mode='markers',
                name=f"{key} (Hist)",
                marker=dict(color=val['color'], opacity=0.5, size=3)
            ))
            
            # Trend Line (Historical + Future)
            # Create full range ordinals for smooth line
            full_dates_ord = np.concatenate([df['date_ordinal'].values.reshape(-1,1), future_ordinals])
            full_dates_dt = [datetime.date.fromordinal(int(d)) for d in full_dates_ord.flatten()]
            trend_y = models_gen[key].predict(full_dates_ord)
            
            fig_gen.add_trace(go.Scatter(
                x=full_dates_dt,
                y=trend_y,
                mode='lines',
                name=f"{key} (Trend)",
                line=dict(color=val['color'], width=2, dash='dash')
            ))

        fig_gen.update_layout(
            xaxis_title="Year",
            yaxis_title="Generation (GWh)",
            template="plotly_dark",
            hovermode="x unified"
        )
        st.plotly_chart(fig_gen, use_container_width=True)

    with tab2:
        st.subheader("Emissions Forecast (tCO₂e)")
        fig_em = go.Figure()

        for key, val in targets.items():
            # Historical Dots
            fig_em.add_trace(go.Scatter(
                x=df['date'], 
                y=df[val['em_col']], 
                mode='markers',
                name=f"{key} (Hist)",
                marker=dict(color=val['color'], opacity=0.5, size=3)
            ))
            
            # Trend Line
            # We calculate this by taking the Gen Trend -> Applying User Formulas
            full_dates_ord = np.concatenate([df['date_ordinal'].values.reshape(-1,1), future_ordinals])
            full_dates_dt = [datetime.date.fromordinal(int(d)) for d in full_dates_ord.flatten()]
            gen_trend = models_gen[key].predict(full_dates_ord)
            
            # Apply User Formulas
            if key == 'Coal':
                em_trend = (958.45 * gen_trend) - 185539
            elif key == 'Gas CCGT':
                em_trend = (413.21 * gen_trend) + 14159
            elif key == 'Gas OCGT':
                em_trend = (582.73 * gen_trend) - 114.71
                
            fig_em.add_trace(go.Scatter(
                x=full_dates_dt,
                y=em_trend,
                mode='lines',
                name=f"{key} (Trend)",
                line=dict(color=val['color'], width=2, dash='dash')
            ))

        fig_em.update_layout(
            xaxis_title="Year",
            yaxis_title="Emissions (tCO₂e)",
            template="plotly_dark",
            hovermode="x unified"
        )
        st.plotly_chart(fig_em, use_container_width=True)

    with tab3:
        st.subheader("Generation vs Emissions Relationship")
        st.caption("Visualizing the linear relationship equations you provided.")
        
        # Dropdown to select source for scatter plot
        selected_source = st.selectbox("Select Energy Source", list(targets.keys()))
        val = targets[selected_source]
        
        fig_scatter = px.scatter(
            df, 
            x=val['gen_col'], 
            y=val['em_col'], 
            trendline="ols", # This calculates OLS dynamically, useful to compare with user equations
            trendline_color_override="white",
            title=f"{selected_source}: Generation vs Emissions"
        )
        
        fig_scatter.update_layout(template="plotly_dark")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Display the equation used
        st.markdown("#### Equation Used for Calculation:")
        if selected_source == 'Coal':
            st.latex(r"y = 958.45x - 185539")
        elif selected_source == 'Gas CCGT':
            st.latex(r"y = 413.21x + 14159")
        elif selected_source == 'Gas OCGT':
            st.latex(r"y = 582.73x - 114.71")
        st.caption("Where x = Generation (GWh) and y = Emissions (tCO₂e)")
