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

# --- Data Loading Logic ---
@st.cache_data
def load_local_data():
    """Attempts to load data from standard local paths."""
    possible_files = [
        "predictor_csv.csv", 
        "predictor_csv.xlsx - Sheet1.csv",
        "predictor_csv.xlsx - Sheet1 (1).csv"
    ]
    
    for file_path in possible_files:
        try:
            df = pd.read_csv(file_path)
            return df
        except FileNotFoundError:
            continue
    return None

def process_dataframe(df):
    """Cleans and prepares the dataframe for analysis."""
    try:
        df['date'] = pd.to_datetime(df['date'])
        df['date_ordinal'] = df['date'].apply(lambda x: x.toordinal())
        return df
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

# --- Main Initialization ---

# Auto-load data directly (Uploader removed as requested)
raw_df = load_local_data()

if raw_df is None:
    st.error("⚠️ Data file not found automatically. Please ensure 'predictor_csv.csv' is in the directory.")
    st.stop()

df = process_dataframe(raw_df)

if df is not None:
    # --- Sidebar Controls ---
    st.sidebar.header("Prediction Controls")
    target_year = st.sidebar.slider("Select Target Year", 2025, 2070, 2030)
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Note on Logic:**\n\n"
        "1. **Date vs Gen:** Uses Python Linear Regression on historical data.\n"
        "2. **Gen vs Emission:** Uses your specified Excel formulas."
    )

    # --- Regression Logic ---
    targets = {
        'Coal': {
            'gen_col': 'Coal (Black) -  GWh',
            'em_col': 'Coal (Black) Emissions Vol - tCO₂e',
            'color': '#000080' # Navy Blue
        },
        'Combined Cycle Gas Turbine': {
            'gen_col': 'Gas (CCGT) -  GWh',
            'em_col': 'Gas (CCGT) Emissions Vol - tCO₂e',
            'color': '#008000' # Green
        },
        'Open Cycle Gas Turbine': {
            'gen_col': 'Gas (OCGT) -  GWh',
            'em_col': 'Gas (OCGT) Emissions Vol - tCO₂e',
            'color': '#800080' # Purple
        }
    }

    # Train Models: Date (Independent) -> Generation (Dependent)
    models_gen = {}
    X = df[['date_ordinal']]
    
    for key, val in targets.items():
        if val['gen_col'] in df.columns:
            y = df[val['gen_col']]
            model = LinearRegression()
            model.fit(X, y)
            models_gen[key] = model
        else:
            st.error(f"Column '{val['gen_col']}' not found in dataset.")
            st.stop()

    # --- Prediction Function ---
    def predict_values(year):
        target_date = datetime.date(year, 6, 1) 
        target_ordinal = target_date.toordinal()
        
        predictions = {}
        total_emissions = 0
        total_gen = 0
        
        for key, val in targets.items():
            gen_pred = models_gen[key].predict([[target_ordinal]])[0]
            
            # User Provided Equations
            if key == 'Coal':
                em_pred = (958.45 * gen_pred) - 185539
            elif key == 'Combined Cycle Gas Turbine':
                em_pred = (413.21 * gen_pred) + 14159
            elif key == 'Open Cycle Gas Turbine':
                em_pred = (582.73 * gen_pred) - 114.71
            
            if gen_pred < 0: gen_pred = 0
            if em_pred < 0: em_pred = 0
            
            predictions[key] = {
                'gen': gen_pred,
                'em': em_pred
            }
            
            total_gen += gen_pred
            total_emissions += em_pred
            
        return predictions, total_gen, total_emissions

    preds, tot_gen, tot_em = predict_values(target_year)

    # --- Dashboard Layout ---
    st.title("NSW Energy and Emissions Forecast Dashboard")
    st.markdown(f"### Projections for Year: **{target_year}**")

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
    
    c1, c2, c3 = st.columns(3)
    sources = ['Coal', 'Combined Cycle Gas Turbine', 'Open Cycle Gas Turbine']
    cols = [c1, c2, c3]
    
    for source, col in zip(sources, cols):
        with col:
            st.markdown(f"#### {source}")
            st.metric("Generation (GWh)", f"{preds[source]['gen']:,.2f}")
            st.metric("Emissions (tCO₂e)", f"{preds[source]['em']:,.2f}")

    st.divider()

    # --- Tabs for Analysis ---
    tab1, tab2, tab3 = st.tabs(["📉 Generation Trends", "☁️ Emission Trends", "📊 Linear Regression Validation"])

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
                marker=dict(color=val['color'], opacity=0.5, size=3),
                hovertemplate=f"<b>{key} (Hist)</b><br>Date: %{{x|%d/%m/%Y}}<br>Gen: %{{y:.4r}} GWh<extra></extra>"
            ))
            
            # Trend Line (Historical + Future)
            full_dates_ord = np.concatenate([df['date_ordinal'].values.reshape(-1,1), future_ordinals])
            full_dates_dt = [datetime.date.fromordinal(int(d)) for d in full_dates_ord.flatten()]
            trend_y = models_gen[key].predict(full_dates_ord)
            
            fig_gen.add_trace(go.Scatter(
                x=full_dates_dt,
                y=trend_y,
                mode='lines',
                name=f"{key} (Trend)",
                line=dict(color=val['color'], width=2, dash='dash'),
                hovertemplate=f"<b>{key} (Trend)</b><br>Year: %{{x|%Y}}<br>Gen: %{{y:.4r}} GWh<extra></extra>"
            ))

        fig_gen.update_layout(
            xaxis_title="Year",
            yaxis_title="Generation (GWh)",
            template="plotly_dark",
            hovermode="x unified",
            hoverlabel=dict(namelength=-1), # Ensures full name visibility
            margin=dict(l=60, r=40, t=40, b=60) # Adjust margins
        )
        fig_gen.update_xaxes(automargin=True)
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
                marker=dict(color=val['color'], opacity=0.5, size=3),
                hovertemplate=f"<b>{key} (Hist)</b><br>Date: %{{x|%d/%m/%Y}}<br>Emissions: %{{y:.4r}} tCO₂e<extra></extra>"
            ))
            
            # Trend Line
            full_dates_ord = np.concatenate([df['date_ordinal'].values.reshape(-1,1), future_ordinals])
            full_dates_dt = [datetime.date.fromordinal(int(d)) for d in full_dates_ord.flatten()]
            gen_trend = models_gen[key].predict(full_dates_ord)
            
            # Apply User Formulas
            if key == 'Coal':
                em_trend = (958.45 * gen_trend) - 185539
            elif key == 'Combined Cycle Gas Turbine':
                em_trend = (413.21 * gen_trend) + 14159
            elif key == 'Open Cycle Gas Turbine':
                em_trend = (582.73 * gen_trend) - 114.71
                
            fig_em.add_trace(go.Scatter(
                x=full_dates_dt,
                y=em_trend,
                mode='lines',
                name=f"{key} (Trend)",
                line=dict(color=val['color'], width=2, dash='dash'),
                hovertemplate=f"<b>{key} (Trend)</b><br>Year: %{{x|%Y}}<br>Emissions: %{{y:.4r}} tCO₂e<extra></extra>"
            ))

        fig_em.update_layout(
            xaxis_title="Year",
            yaxis_title="Emissions (tCO₂e)",
            template="plotly_dark",
            hovermode="x unified",
            hoverlabel=dict(namelength=-1), # Ensures full name visibility
            margin=dict(l=60, r=40, t=40, b=60)
        )
        fig_em.update_xaxes(automargin=True)
        st.plotly_chart(fig_em, use_container_width=True)

    with tab3:
        st.subheader("Generation vs Emissions Relationship")
        st.caption("Visualizing the linear relationship equations you provided.")
        
        selected_source = st.selectbox("Select Energy Source", list(targets.keys()))
        val = targets[selected_source]
        
        # Original Data Scatter
        fig_scatter = px.scatter(
            df, 
            x=val['gen_col'], 
            y=val['em_col'], 
            title=f"{selected_source}: Generation vs Emissions"
        )
        
        # Manually Add Trend Line based on User Equation
        x_min, x_max = df[val['gen_col']].min(), df[val['gen_col']].max()
        x_range = np.linspace(x_min, x_max, 100)
        
        equation_str = ""
        if selected_source == 'Coal':
            y_trend = 958.45 * x_range - 185539
            equation_str = "y = 958.45x - 185539"
        elif selected_source == 'Combined Cycle Gas Turbine':
            y_trend = 413.21 * x_range + 14159
            equation_str = "y = 413.21x + 14159"
        elif selected_source == 'Open Cycle Gas Turbine':
            y_trend = 582.73 * x_range - 114.71
            equation_str = "y = 582.73x - 114.71"
            
        fig_scatter.add_trace(go.Scatter(
            x=x_range,
            y=y_trend,
            mode='lines',
            name=f'Eq: {equation_str}',
            line=dict(color=val['color'], width=2),
            hovertemplate="<b>Trend</b><br>Gen: %{x:.4r}<br>Emissions: %{y:.4r}<extra></extra>"
        ))
        
        # Update markers in the scatter to use 4 sig figs
        fig_scatter.update_traces(
            selector=dict(mode='markers'),
            hovertemplate="<b>Data</b><br>Gen: %{x:.4r}<br>Emissions: %{y:.4r}<extra></extra>"
        )
        
        fig_scatter.update_layout(
            template="plotly_dark",
            hoverlabel=dict(namelength=-1)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.markdown("#### Equation Used:")
        st.latex(equation_str)
        st.caption("Where x = Generation (GWh) and y = Emissions (tCO₂e)")
