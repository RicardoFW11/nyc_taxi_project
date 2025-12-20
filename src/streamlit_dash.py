import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="NYC Taxi Fare Analysis",
    page_icon="hk-taxi",
    layout="wide"
)

st.title("🚖 NYC Yellow Taxi Fare Dashboard")
st.markdown("""
This dashboard analyzes the **Fare Amount** problematic, exploring outliers, 
correlations, and temporal patterns in the NYC Yellow Taxi dataset.
""")

# --- 2. DATA LOADING FUNCTION ---
@st.cache_data
def load_data():
    """
    Loads the dataset. 
    Tries to load the parquet file first (fast), then falls back to CSV.
    """
    # Adjust these paths to match your actual project structure
    # Based on your README, data is in 'data/'
    parquet_path = "data/yellow_tripdata_2022-05.parquet" 
    csv_path = "data/raw/taxi_data.csv"
    
    # Try Parquet first (from your teammate's EDA)
    try:
        df = pd.read_parquet(parquet_path)
        # Take a sample if data is huge to keep dashboard fast
        if len(df) > 100000:
            df = df.sample(100000, random_state=42)
    except FileNotFoundError:
        try:
            # Fallback to CSV sample
            df = pd.read_csv(csv_path, nrows=100000)
        except FileNotFoundError:
            st.error("Data file not found. Please ensure 'yellow_tripdata_2022-05.parquet' or 'taxi_data.csv' is in your 'data/' folder.")
            return None

    # --- PREPROCESSING ---
    # Convert dates
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
    
    # Feature Engineering (Temporal)
    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day_name'] = df['tpep_pickup_datetime'].dt.day_name()
    df['trip_duration_min'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    
    # Filter logical errors (Negative fares, 0 distance) based on EDA
    df = df[ (df['fare_amount'] > 0) & (df['trip_distance'] > 0) & (df['trip_duration_min'] > 0) ]
    
    return df

df = load_data()

if df is not None:
    # --- 3. SIDEBAR FILTERS ---
    st.sidebar.header("🎛️ Filters")
    
    # Filter by Fare
    fare_range = st.sidebar.slider("Fare Amount Range ($)", 0, 200, (0, 100))
    
    # Filter by Payment Type
    payment_map = {1: 'Credit Card', 2: 'Cash', 3: 'No Charge', 4: 'Dispute', 5: 'Unknown'}
    df['payment_label'] = df['payment_type'].map(payment_map)
    pay_filter = st.sidebar.multiselect("Payment Type", df['payment_label'].unique(), default=['Credit Card', 'Cash'])
    
    # Apply Filters
    df_filtered = df[
        (df['fare_amount'].between(fare_range[0], fare_range[1])) &
        (df['payment_label'].isin(pay_filter))
    ]

    # --- 4. KPI ROW ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trips", f"{len(df_filtered):,}")
    col2.metric("Avg Fare", f"${df_filtered['fare_amount'].mean():.2f}")
    col3.metric("Avg Distance", f"{df_filtered['trip_distance'].mean():.2f} miles")
    col4.metric("Avg Tip", f"${df_filtered['tip_amount'].mean():.2f}")

    st.markdown("---")

    # --- 5. MAIN VISUALIZATIONS ---
    
    # ROW 1: Distribution & Correlation
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("📊 Fare Distribution")
        fig_hist = px.histogram(
            df_filtered, 
            x="fare_amount", 
            nbins=50, 
            color_discrete_sequence=['#636EFA'],
            title="Histogram of Fare Amounts"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption("Most fares are concentrated in the lower range, with a long tail of expensive rides.")

    with c2:
        st.subheader("🔥 Correlation Heatmap")
        # Compute correlation
        corr_cols = ['fare_amount', 'trip_distance', 'trip_duration_min', 'tip_amount', 'tolls_amount']
        corr_matrix = df_filtered[corr_cols].corr()
        
        # Plot using Seaborn/Matplotlib
        fig_corr, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig_corr)
        st.caption("Strong correlation between Distance and Fare (as expected). Duration is also key.")

    # ROW 2: Temporal Heatmap (Creative)
    st.subheader("⏰ Peak Hours: When do High Fares happen?")
    
    # Pivot for Heatmap: Day vs Hour -> Avg Fare
    heatmap_data = df_filtered.groupby(['day_name', 'hour'])['fare_amount'].mean().reset_index()
    # Sort days
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    fig_heat = px.density_heatmap(
        heatmap_data, 
        x="hour", 
        y="day_name", 
        z="fare_amount", 
        histfunc="avg",
        color_continuous_scale="Viridis",
        category_orders={"day_name": days_order},
        title="Average Fare by Day and Hour"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ROW 3: Scatter Analysis (Fare vs Distance)
    st.subheader("📈 Fare vs. Distance (Detecting Anomalies)")
    
    # Create a scatter plot
    fig_scatter = px.scatter(
        df_filtered.sample(min(5000, len(df_filtered))), # Sample for performance
        x="trip_distance", 
        y="fare_amount", 
        color="payment_label",
        hover_data=['trip_duration_min'],
        opacity=0.6,
        title="Fare Amount vs Trip Distance (Sample of 5000 rides)"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption("Points far above the main line might be traffic or scams. Points on the X-axis are distance with 0 fare.")

else:
    st.warning("Awaiting Data... Please ensure your data file path is correct in the script.")