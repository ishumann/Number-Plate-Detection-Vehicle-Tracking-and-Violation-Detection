import streamlit as st
import pandas as pd
import numpy as np
import cv2
import yaml
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_control.traffic_signal import SignalState

# Page configuration
st.set_page_config(
    page_title="Traffic Control System",
    page_icon="🚦",
    layout="wide"
)

# Load configuration
@st.cache_resource
def load_config():
    with open("config/config.yaml", 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# Sidebar
st.sidebar.title("Traffic Control System")
selected_page = st.sidebar.radio(
    "Navigation",
    ["Live Monitoring", "Violation Reports", "Traffic Analytics", "System Settings"]
)

# Helper functions
def load_recent_violations(n_minutes=30):
    """Load recent violations from CSV file."""
    try:
        df = pd.read_csv('data/violations.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        cutoff_time = datetime.now() - timedelta(minutes=n_minutes)
        return df[df['timestamp'] >= cutoff_time]
    except Exception as e:
        st.error(f"Error loading violations data: {str(e)}")
        return pd.DataFrame()

def load_traffic_data(n_minutes=30):
    """Load recent traffic data from CSV file."""
    try:
        df = pd.read_csv('data/traffic.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        cutoff_time = datetime.now() - timedelta(minutes=n_minutes)
        return df[df['timestamp'] >= cutoff_time]
    except Exception as e:
        st.error(f"Error loading traffic data: {str(e)}")
        return pd.DataFrame()

# Live Monitoring Page
def show_live_monitoring():
    st.title("Live Traffic Monitoring")
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Video feed placeholder
        st.subheader("Live Camera Feed")
        video_placeholder = st.empty()
        
        # Traffic signal status
        st.subheader("Traffic Signal Status")
        signal_cols = st.columns(4)
        for i, col in enumerate(signal_cols, 1):
            with col:
                st.metric(f"Phase {i}", "🔴" if i != 1 else "🟢")
                
    with col2:
        # Current statistics
        st.subheader("Current Statistics")
        
        # Vehicle count
        df_traffic = load_traffic_data(5)  # Last 5 minutes
        if not df_traffic.empty:
            current_count = df_traffic['vehicle_count'].iloc[-1]
            avg_count = df_traffic['vehicle_count'].mean()
            st.metric("Vehicle Count", current_count, 
                     f"{current_count - avg_count:+.0f} from average")
            
        # Recent violations
        df_violations = load_recent_violations(5)  # Last 5 minutes
        if not df_violations.empty:
            st.metric("Recent Violations", len(df_violations))
            
        # Traffic density
        st.subheader("Traffic Density")
        if not df_traffic.empty:
            densities = eval(df_traffic['densities'].iloc[-1])
            fig = go.Figure(data=[
                go.Bar(
                    x=[f"Phase {i}" for i in range(1, 5)],
                    y=[densities[i] for i in range(1, 5)]
                )
            ])
            fig.update_layout(height=200, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)

# Violation Reports Page
def show_violation_reports():
    st.title("Violation Reports")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now().date() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", datetime.now().date())
        
    # Load violations data
    df = pd.read_csv('data/violations.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    
    # Filter by date range
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df_filtered = df.loc[mask]
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Violations", len(df_filtered))
    with col2:
        st.metric("Unique Vehicles", df_filtered['license_plate'].nunique())
    with col3:
        st.metric("Avg. Violations/Day", 
                 f"{len(df_filtered) / max(1, (end_date - start_date).days):.1f}")
    
    # Violations by type
    st.subheader("Violations by Type")
    fig = px.pie(df_filtered, names='type', title="Distribution of Violations")
    st.plotly_chart(fig)
    
    # Violations over time
    st.subheader("Violations Over Time")
    daily_violations = df_filtered.groupby('date').size().reset_index(name='count')
    fig = px.line(daily_violations, x='date', y='count', title="Daily Violations")
    st.plotly_chart(fig)
    
    # Detailed violation list
    st.subheader("Detailed Violation List")
    st.dataframe(
        df_filtered[['timestamp', 'type', 'vehicle_class', 'license_plate', 'speed']]
        .sort_values('timestamp', ascending=False)
    )

# Traffic Analytics Page
def show_traffic_analytics():
    st.title("Traffic Analytics")
    
    # Time range selector
    time_range = st.selectbox(
        "Time Range",
        ["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days"]
    )
    
    # Load and process traffic data
    df = pd.read_csv('data/traffic.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter based on selected time range
    now = datetime.now()
    if time_range == "Last Hour":
        df = df[df['timestamp'] >= now - timedelta(hours=1)]
    elif time_range == "Last 24 Hours":
        df = df[df['timestamp'] >= now - timedelta(days=1)]
    elif time_range == "Last 7 Days":
        df = df[df['timestamp'] >= now - timedelta(days=7)]
    else:
        df = df[df['timestamp'] >= now - timedelta(days=30)]
    
    # Traffic volume over time
    st.subheader("Traffic Volume Over Time")
    fig = px.line(df, x='timestamp', y='vehicle_count', title="Vehicle Count")
    st.plotly_chart(fig)
    
    # Average traffic density by phase
    st.subheader("Average Traffic Density by Phase")
    densities_df = pd.DataFrame([eval(d) for d in df['densities']])
    avg_densities = densities_df.mean()
    fig = go.Figure(data=[
        go.Bar(
            x=[f"Phase {i}" for i in range(1, 5)],
            y=avg_densities
        )
    ])
    st.plotly_chart(fig)
    
    # Peak hours analysis
    st.subheader("Peak Hours Analysis")
    df['hour'] = df['timestamp'].dt.hour
    hourly_avg = df.groupby('hour')['vehicle_count'].mean().reset_index()
    fig = px.bar(hourly_avg, x='hour', y='vehicle_count',
                 title="Average Vehicle Count by Hour")
    st.plotly_chart(fig)

# System Settings Page
def show_system_settings():
    st.title("System Settings")
    
    # Load current configuration
    config = load_config()
    
    # Camera settings
    st.subheader("Camera Settings")
    col1, col2 = st.columns(2)
    with col1:
        resolution_w = st.number_input("Resolution Width", 
                                     value=config['camera']['resolution'][0])
    with col2:
        resolution_h = st.number_input("Resolution Height",
                                     value=config['camera']['resolution'][1])
    fps = st.slider("FPS", min_value=1, max_value=60, 
                    value=config['camera']['fps'])
    
    # Detection settings
    st.subheader("Detection Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=config['detection']['confidence_threshold']
    )
    
    # Traffic signal settings
    st.subheader("Traffic Signal Settings")
    col1, col2 = st.columns(2)
    with col1:
        min_green = st.number_input(
            "Minimum Green Time (seconds)",
            value=config['signal']['min_green_time']
        )
    with col2:
        max_green = st.number_input(
            "Maximum Green Time (seconds)",
            value=config['signal']['max_green_time']
        )
    
    # Save changes button
    if st.button("Save Changes"):
        # Update configuration
        config['camera']['resolution'] = [int(resolution_w), int(resolution_h)]
        config['camera']['fps'] = int(fps)
        config['detection']['confidence_threshold'] = float(confidence_threshold)
        config['signal']['min_green_time'] = int(min_green)
        config['signal']['max_green_time'] = int(max_green)
        
        # Save to file
        try:
            with open("config/config.yaml", 'w') as f:
                yaml.dump(config, f)
            st.success("Settings saved successfully!")
        except Exception as e:
            st.error(f"Error saving settings: {str(e)}")

# Display selected page
if selected_page == "Live Monitoring":
    show_live_monitoring()
elif selected_page == "Violation Reports":
    show_violation_reports()
elif selected_page == "Traffic Analytics":
    show_traffic_analytics()
else:
    show_system_settings() 