import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os

# ---------- Page Config ----------
st.set_page_config(page_title="Afficionado Coffee Roasters - Sales Trends", layout="wide")

# ---------- Custom CSS: Warm Coffee Background ----------
st.markdown("""
<style>
    /* Main background - latte to caramel gradient */
    .stApp {
        background: linear-gradient(145deg, #f5e6d3 0%, #d4b48c 40%, #b8956a 100%);
    }
    /* Sidebar - warm glass effect */
    .css-1d391kg, .css-12oz5g7 {
        background: rgba(245, 230, 211, 0.75) !important;
        backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(160, 120, 80, 0.3);
    }
    /* All text - dark brown for contrast */
    h1, h2, h3, h4, h5, h6, label, .stMarkdown, p, div {
        color: #3e2723 !important;
    }
    /* Metrics - warm white with shadow */
    .stMetric {
        background: rgba(255, 248, 240, 0.7);
        border: 1px solid #b8956a;
        border-radius: 12px;
        padding: 12px;
        backdrop-filter: blur(4px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .stMetric label {
        color: #5a3d2b !important;
    }
    .stMetric .stMetricValue {
        color: #3e2723 !important;
    }
    /* Tabs - coffee bean accents */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 248, 240, 0.5);
        border-radius: 12px;
        padding: 6px;
        backdrop-filter: blur(4px);
        border: 1px solid #b8956a;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        background: transparent;
        color: #5a3d2b;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #8b6b4b !important;
        color: #f5e6d3 !important;
        box-shadow: 0 2px 8px rgba(139, 107, 75, 0.4);
    }
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 248, 240, 0.5) !important;
        backdrop-filter: blur(4px);
        border: 1px solid #b8956a;
        border-radius: 8px;
        color: #3e2723 !important;
    }
    /* Buttons */
    .stButton button {
        background: #8b6b4b;
        color: #f5e6d3;
        border: 1px solid #b8956a;
    }
    /* Dataframe - light theme */
    .dataframe {
        background: rgba(255, 248, 240, 0.8) !important;
        color: #3e2723 !important;
    }
    .dataframe th {
        background: #d4b48c !important;
        color: #3e2723 !important;
    }
    .dataframe td {
        border-color: #b8956a !important;
    }
    /* Scrollbar - coffee tones */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f5e6d3;
    }
    ::-webkit-scrollbar-thumb {
        background: #b8956a;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #a07050;
    }
</style>
""", unsafe_allow_html=True)
# ---------- Data Loading and Preprocessing ----------
@st.cache_data
def load_data_from_file(file_path):
    df = pd.read_csv("Afficionado Coffee Roasters - Transactions.csv")
    return preprocess_data(df)

@st.cache_data
def load_data_from_uploaded(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return preprocess_data(df)

def preprocess_data(df):
    # Basic validation (fixed)
    assert df['transaction_id'].is_unique, "Duplicate transaction IDs found"
    assert (df['transaction_qty'] > 0).all(), "Negative or zero quantity found"
    assert (df['unit_price'] > 0).all(), "Negative or zero unit price found"
    # Parse transaction_time to datetime
    df['transaction_time'] = pd.to_datetime(df['transaction_time'], format='%H:%M:%S')
    df['transaction_datetime'] = pd.to_datetime(
        '2025-01-01 ' + df['transaction_time'].dt.strftime('%H:%M:%S')
    )
    # Compute revenue
    df['revenue'] = df['transaction_qty'] * df['unit_price']
    # Feature engineering
    df['hour'] = df['transaction_time'].dt.hour
    df['day_of_week'] = df['transaction_datetime'].dt.day_name()
    # Order days
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=day_order, ordered=True)
    # Time bucket
    def time_bucket(hour):
        if 6 <= hour <= 11:
            return 'Morning (6-11)'
        elif 12 <= hour <= 16:
            return 'Afternoon (12-16)'
        elif 17 <= hour <= 21:
            return 'Evening (17-21)'
        else:
            return 'Late (22-5)'
    df['time_bucket'] = df['hour'].apply(time_bucket)
    return df

# ---------- Analysis Functions ----------
def daily_revenue_trend(df):
    daily = df.groupby(df['transaction_datetime'].dt.date).agg(
        total_revenue=('revenue', 'sum'),
        transaction_count=('transaction_id', 'count')
    ).reset_index()
    daily['date'] = pd.to_datetime(daily['transaction_datetime'])
    return daily

def weekly_aggregation(df):
    df['week'] = df['transaction_datetime'].dt.isocalendar().week
    weekly = df.groupby('week').agg(
        total_revenue=('revenue', 'sum'),
        transaction_count=('transaction_id', 'count')
    ).reset_index()
    return weekly

def day_of_week_performance(df):
    dow = df.groupby('day_of_week').agg(
        avg_revenue=('revenue', 'mean'),
        avg_transactions=('transaction_id', 'count')
    ).reset_index()
    # Count unique dates per day of week using apply
    days_count = df.groupby('day_of_week')['transaction_datetime'].apply(
        lambda x: x.dt.date.nunique()
    ).reset_index(name='num_days')
    dow = dow.merge(days_count, on='day_of_week')
    dow['avg_daily_transactions'] = dow['avg_transactions'] / dow['num_days']
    return dow

def hourly_demand(df):
    hourly = df.groupby('hour').agg(
        total_revenue=('revenue', 'sum'),
        transaction_count=('transaction_id', 'count')
    ).reset_index()
    return hourly

def location_temporal_comparison(df):
    loc_hourly = df.groupby(['store_location', 'hour']).agg(
        total_revenue=('revenue', 'sum'),
        transaction_count=('transaction_id', 'count')
    ).reset_index()
    return loc_hourly

# ---------- Streamlit App ----------
def main():
    st.set_page_config(page_title="Afficionado Coffee Roasters - Sales Trends", layout="wide")
    st.title("☕ Afficionado Coffee Roasters - Sales Trend and Time-Based Performance Analysis")
    st.markdown("### Explore sales patterns across time and locations")

    # ---- Sidebar: CSV Path Input ----
    st.sidebar.header("Data Source")
    default_path = "Afficionado Coffee Roasters - Transactions.csv"
    csv_path = st.sidebar.text_input("CSV file path", value=default_path)

    # ---- Load Data ----
    df = None
    if csv_path and os.path.exists(csv_path):
        try:
            df = load_data_from_file(csv_path)
            st.sidebar.success(f"Loaded from {csv_path}")
        except Exception as e:
            st.sidebar.warning(f"Could not load from path: {e}")
    
    if df is None:
        st.info("Please upload the CSV file or provide a valid file path.")
        uploaded_file = st.file_uploader("Upload your transactions CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                df = load_data_from_uploaded(uploaded_file)
                st.success("Data loaded successfully from upload!")
            except Exception as e:
                st.error(f"Error loading uploaded file: {e}")
                st.stop()
        else:
            st.stop()

    # ---- Sidebar Filters ----
    st.sidebar.header("Filters")
    locations = ['All'] + sorted(df['store_location'].unique())
    selected_location = st.sidebar.selectbox("Store Location", locations)

    days = ['All'] + list(df['day_of_week'].cat.categories)
    selected_days = st.sidebar.multiselect("Day of Week", days, default=['All'])

    hour_min, hour_max = st.sidebar.slider("Hour Range", 0, 23, (0, 23))

    filtered_df = df.copy()
    if selected_location != 'All':
        filtered_df = filtered_df[filtered_df['store_location'] == selected_location]
    if 'All' not in selected_days:
        filtered_df = filtered_df[filtered_df['day_of_week'].isin(selected_days)]
    filtered_df = filtered_df[(filtered_df['hour'] >= hour_min) & (filtered_df['hour'] <= hour_max)]

    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please adjust.")
        return

    # ---- Metrics ----
    total_revenue = filtered_df['revenue'].sum()
    total_transactions = filtered_df['transaction_id'].count()
    avg_transaction_value = total_revenue / total_transactions if total_transactions else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"${total_revenue:,.2f}")
    col2.metric("Total Transactions", f"{total_transactions:,}")
    col3.metric("Avg Transaction Value", f"${avg_transaction_value:.2f}")

    # ---- Tabs ----
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Sales Trends", "📊 Day of Week", "⏰ Hourly Demand", "📍 Location Comparison"])

    with tab1:
        st.header("Sales Trends Over Time")
        daily = daily_revenue_trend(filtered_df)
        fig_daily = px.line(daily, x='date', y='total_revenue', title='Daily Revenue Trend')
        st.plotly_chart(fig_daily, use_container_width=True)

        weekly = weekly_aggregation(filtered_df)
        fig_weekly = px.bar(weekly, x='week', y='total_revenue', title='Weekly Revenue Aggregation')
        st.plotly_chart(fig_weekly, use_container_width=True)

    with tab2:
        st.header("Day of Week Performance")
        dow = day_of_week_performance(filtered_df)
        fig_dow_rev = px.bar(dow, x='day_of_week', y='avg_revenue', title='Average Revenue per Day (per transaction)')
        st.plotly_chart(fig_dow_rev, use_container_width=True)

        fig_dow_trans = px.bar(dow, x='day_of_week', y='avg_daily_transactions', title='Average Daily Transactions')
        st.plotly_chart(fig_dow_trans, use_container_width=True)

    with tab3:
        st.header("Hourly Demand Analysis")
        hourly = hourly_demand(filtered_df)
        fig_hour_rev = px.bar(hourly, x='hour', y='total_revenue', title='Total Revenue by Hour')
        st.plotly_chart(fig_hour_rev, use_container_width=True)

        fig_hour_trans = px.bar(hourly, x='hour', y='transaction_count', title='Transaction Count by Hour')
        st.plotly_chart(fig_hour_trans, use_container_width=True)

        st.subheader("Hourly Heatmap (Revenue)")
        pivot = filtered_df.pivot_table(index='day_of_week', columns='hour', values='revenue', aggfunc='sum', fill_value=0)
        pivot = pivot.reindex(df['day_of_week'].cat.categories)
        fig_heat = px.imshow(pivot, text_auto=True, aspect="auto", title="Revenue Heatmap by Day and Hour")
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab4:
        st.header("Location Comparison")
        if selected_location == 'All':
            loc_hourly = location_temporal_comparison(filtered_df)
            fig_loc = px.line(loc_hourly, x='hour', y='total_revenue', color='store_location',
                              title='Hourly Revenue by Store Location')
            st.plotly_chart(fig_loc, use_container_width=True)

            loc_total = filtered_df.groupby('store_location')['revenue'].sum().reset_index()
            fig_loc_total = px.bar(loc_total, x='store_location', y='revenue', title='Total Revenue by Location')
            st.plotly_chart(fig_loc_total, use_container_width=True)
        else:
            st.info("Select 'All' in the store filter to compare locations.")

    # Raw data preview
    with st.expander("View Raw Data (Filtered)"):
        st.dataframe(filtered_df.head(100))

if __name__ == "__main__":
    main()